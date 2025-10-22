# MIT License

# Copyright (c) 2025 Tonghe Zhang

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import os
from typing import Optional, Sequence
import numpy as np

from mplib import Planner
from mplib.pymp import ArticulatedModel, PlanningWorld
from mplib.pymp.planning import ompl

class NonConvexPlanner(Planner):
    """Motion planner"""

    # constructor ankor
    def __init__(
        self,
        urdf: str,
        move_group: str,
        srdf: str = "",
        package_keyword_replacement: str = "",
        user_link_names: Sequence[str] = [],
        user_joint_names: Sequence[str] = [],
        joint_vel_limits: Optional[Sequence[float] | np.ndarray] = None,
        joint_acc_limits: Optional[Sequence[float] | np.ndarray] = None,
        **kwargs,
    ):
        if joint_vel_limits is None:
            joint_vel_limits = []
        if joint_acc_limits is None:
            joint_acc_limits = []
        self.urdf = urdf
        if srdf == "" and os.path.exists(urdf.replace(".urdf", ".srdf")):
            self.srdf = urdf.replace(".urdf", ".srdf")
            print(f"No SRDF file provided but found {self.srdf}")

        # replace package:// keyword if exists
        urdf = self.replace_package_keyword(package_keyword_replacement)

        self.robot = ArticulatedModel(
            urdf,
            srdf,
            [0, 0, -9.81],
            user_link_names,
            user_joint_names,
            convex=False,
            verbose=False,
        )
        self.pinocchio_model = self.robot.get_pinocchio_model()
        self.user_link_names = self.pinocchio_model.get_link_names()
        self.user_joint_names = self.pinocchio_model.get_joint_names()

        self.planning_world = PlanningWorld(
            [self.robot],
            ["robot"],
            kwargs.get("normal_objects", []),
            kwargs.get("normal_object_names", []),
        )

        if srdf == "":
            self.generate_collision_pair()
            self.robot.update_SRDF(self.srdf)

        self.joint_name_2_idx = {}
        for i, joint in enumerate(self.user_joint_names):
            self.joint_name_2_idx[joint] = i
        self.link_name_2_idx = {}
        for i, link in enumerate(self.user_link_names):
            self.link_name_2_idx[link] = i

        assert (
            move_group in self.user_link_names
        ), f"end-effector not found as one of the links in {self.user_link_names}"
        self.move_group = move_group
        self.robot.set_move_group(self.move_group)
        self.move_group_joint_indices = self.robot.get_move_group_joint_indices()

        self.joint_types = self.pinocchio_model.get_joint_types()
        self.joint_limits = np.concatenate(self.pinocchio_model.get_joint_limits())
        self.joint_vel_limits = (
            joint_vel_limits
            if len(joint_vel_limits)
            else np.ones(len(self.move_group_joint_indices))
        )
        self.joint_acc_limits = (
            joint_acc_limits
            if len(joint_acc_limits)
            else np.ones(len(self.move_group_joint_indices))
        )
        self.move_group_link_id = self.link_name_2_idx[self.move_group]
        assert len(self.joint_vel_limits) == len(self.joint_acc_limits), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) =/= "
            f"length of joint_acc_limits ({len(self.joint_acc_limits)})"
        )
        assert len(self.joint_vel_limits) == len(self.move_group_joint_indices), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) =/= "
            f"length of move_group ({len(self.move_group_joint_indices)})"
        )
        assert len(self.joint_vel_limits) <= len(self.joint_limits), (
            f"length of joint_vel_limits ({len(self.joint_vel_limits)}) > "
            f"number of total joints ({len(self.joint_limits)})"
        )

        self.planning_world = PlanningWorld([self.robot], ["robot"], [], [])
        self.planner = ompl.OMPLPlanner(world=self.planning_world)
        