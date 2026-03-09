"""
Bimanual YAM arms Inverse Kinematics Example using PyRoki with ViserAbstractBase.
"""

from copy import deepcopy
from typing import Literal, Optional

import numpy as np
import pyroki as pk
import viser, time
import viser.extras
import viser.transforms as vtf

from yam_realtime.robots.inverse_kinematics.pyroki_snippets._solve_ik import solve_ik
from yam_realtime.robots.viser.viser_base import TransformHandle, ViserAbstractBase


class YamPyroki(ViserAbstractBase):
    """
    YAM robot visualization using PyRoki for inverse kinematics.
    Enhanced with coordinate frame support, TCP offset controls, and performance optimizations.
    """

    def __init__(
        self,
        rate: float = 100.0,
        viser_server: Optional[viser.ViserServer] = None,
        bimanual: bool = False,
        coordinate_frame: Literal["base", "world"] = "base",
    ):
        self.robot: Optional[pk.Robot] = None
        self.target_link_names = ["link_6"]
        self.joints = {"left": np.zeros(6)}
        self.coordinate_frame = coordinate_frame
        self.has_jitted_left = False
        self.has_jitted_right = False

        if bimanual:
            self.target_link_names = self.target_link_names * 2
            self.joints["right"] = np.zeros(6)
        super().__init__(rate, viser_server, bimanual=bimanual)

    def _setup_visualization(self):
        super()._setup_visualization()
        if self.bimanual:
            self.base_frame_right = self.viser_server.scene.add_frame("/base/base_right", show_axes=False)
            self.base_frame_right.position = (0.0, -0.61, 0.0)
            self.urdf_vis_right = viser.extras.ViserUrdf(
                self.viser_server, self.urdf, root_node_name="/base/base_right"
            )

    def _setup_solver_specific(self):
        """Setup PyRoki-specific components."""
        self.robot = pk.Robot.from_urdf(self.urdf)
        self.rest_pose = self.urdf.cfg
        
        # Warm up JIT compilation for both arms to avoid first-call latency
        print("Warming up IK solver JIT compilation...")
        dummy_position = np.array([0.12, 0.0, 0.1])
        dummy_wxyz = np.array([0.5, 0.5, 0.5, 0.5])
        
        # Warm up left arm
        _ = solve_ik(
            robot=self.robot,
            target_link_name=self.target_link_names[0],
            target_position=dummy_position,
            target_wxyz=dummy_wxyz,
        )
        self.has_jitted_left = True
        print("  Left arm IK JIT compiled.")
        
        # Warm up right arm (same link but ensures full JIT warmup)
        if self.bimanual:
            _ = solve_ik(
                robot=self.robot,
                target_link_name=self.target_link_names[1],
                target_position=dummy_position,
                target_wxyz=dummy_wxyz,
            )
            self.has_jitted_right = True
            print("  Right arm IK JIT compiled.")
        
        print("IK solver JIT warmup complete.")

    def _setup_gui(self):
        """Setup GUI elements."""
        super()._setup_gui()

        # Add timing displays for each arm
        self.timing_handle_left = self.viser_server.gui.add_number("Left Arm Time (ms)", 0.01, disabled=True)
        if self.bimanual:
            self.timing_handle_right = self.viser_server.gui.add_number("Right Arm Time (ms)", 0.01, disabled=True)

    def _initialize_transform_handles(self):
        """Initialize transform handle positions for arm IK targets."""
        if self.transform_handles["left"].control is not None:
            # self.transform_handles["left"].control.position = (0.25, 0.0, 0.26)
            # self.transform_handles["left"].control.wxyz = vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz
            # self.transform_handles["left"].tcp_offset_frame.position = (
            #     0.0,
            #     0.04,
            #     -0.13,
            # )  # YAM gripper end is slightly offset from the end of the link_6
            self.transform_handles["left"].control.position = (0.12, 0.00535176, 0.09107439)
            self.transform_handles["left"].control.wxyz = (0.5, 0.5, 0.5, 0.5)
            self.transform_handles["left"].tcp_offset_frame.position = (
                0.0,
                0.0,
                0.0,
            )  # YAM gripper end is slightly offset from the end of the link_6

        if self.bimanual:
            if self.transform_handles["right"].control is not None:
                self.transform_handles["right"].control.remove()
                self.transform_handles["right"].tcp_offset_frame.remove()
            self.transform_handles["right"] = TransformHandle(
                tcp_offset_frame=self.viser_server.scene.add_frame(
                    "/base/base_righttarget_right/tcp_offset",
                    0.0,
                    0.0,
                    0.0,
                ),
                control=self.viser_server.scene.add_transform_controls(
                    "/base/base_right/target_right",
                    scale=self.tf_size_handle.value,
                    # position=(0.25, 0.0, 0.26),
                    # wxyz=vtf.SO3.from_rpy_radians(np.pi / 2, 0.0, np.pi / 2).wxyz,
                    position=(0.12, 0.00535176, 0.09107439),
                    wxyz=(0.5, 0.5, 0.5, 0.5),
                ),
            )

    def _update_optional_handle_sizes(self):
        """Update optional handle sizes (none for this implementation)."""
        pass

    def get_target_poses(self):
        """Get target poses with optional TCP offset applied."""
        target_poses = {}

        for side, handle in self.transform_handles.items():
            if handle.control is None:
                continue

            # Combine control handle with TCP offset
            control_tf = vtf.SE3(np.array([*handle.control.wxyz, *handle.control.position]))
            tcp_offset_tf = vtf.SE3(np.array([*handle.tcp_offset_frame.wxyz, *handle.tcp_offset_frame.position]))
            target_poses[side] = control_tf @ tcp_offset_tf

        return target_poses

    def solve_ik(self):
        """Solve inverse kinematics for arm IK targets."""
        if self.robot is None:
            return

        target_poses = self.get_target_poses()

        if self.bimanual:
            if "left" not in target_poses or "right" not in target_poses:
                return
        elif "left" not in target_poses:
            return

        target_positions = []
        target_wxyzs = []
        for idx, side in enumerate(self.get_target_poses().keys()):
            target_tf = target_poses[side]
            target_positions.append(target_tf.translation())
            target_wxyzs.append(target_tf.rotation().wxyz)

            solution = solve_ik(
                robot=self.robot,
                target_link_name=self.target_link_names[idx],
                target_position=target_tf.translation(),
                target_wxyz=target_tf.rotation().wxyz,
            )
            self.joints[side] = solution

    def update_visualization(self):
        """Update visualization with current joint configurations."""
        if self.joints is not None:
            # self.print_current_poses() # print coordinates of the current pose
            self.urdf_vis_left.update_cfg(self.joints["left"])
            if self.bimanual:
                self.urdf_vis_right.update_cfg(self.joints["right"])

    def home(self):
        """Reset both arms to rest pose."""
        self.joints["left"] = self.rest_pose.copy()
        if self.bimanual:
            self.joints["right"] = self.rest_pose.copy()

        self._initialize_transform_handles()

        self.urdf_vis_left.update_cfg(self.rest_pose)
        if self.bimanual:
            self.urdf_vis_right.update_cfg(self.rest_pose)

    def get_joint_positions(self) -> Optional[np.ndarray]:
        """Get current joint positions for the bimanual robot."""
        if self.bimanual:
            if self.joints["left"] is not None and self.joints["right"] is not None:
                return np.concatenate([self.joints["left"], self.joints["right"]])
            else:
                return None
        elif self.joints["left"] is not None:
            return self.joints["left"]

    # Convenience methods for unified interface
    def solve_ik_world(self, target_positions, target_wxyzs=None):
        """Convenience method for IK with world coordinate targets."""
        return self.solve_ik_with_targets(target_positions, target_wxyzs, coordinate_frame="world")

    def solve_ik_base(self, target_positions, target_wxyzs=None):
        """Convenience method for IK with base coordinate targets."""
        return self.solve_ik_with_targets(target_positions, target_wxyzs, coordinate_frame="base")

    # left and right arm are flipped
    # def print_current_poses(self):
    #     poses = self.get_target_poses()
    #     for side, pose in poses.items():
    #         pos = pose.translation()
    #         rot = pose.rotation().wxyz
    #         print(f"{side.capitalize()} Arm Pose:")
    #         print(f"  Position: {pos}")
    #         print(f"  Rotation (wxyz): {rot}")


######################################################## Oculus Quest ########################################################
    # def set_target_pose(self, side: str, se3_pose: vtf.SE3):
    #     if side not in self.transform_handles:
    #         raise ValueError(f"Invalid side: {side}")
    #     if self.transform_handles[side].control is not None:
    #         self.transform_handles[side].control.position = se3_pose.translation()
    #         self.transform_handles[side].control.wxyz = se3_pose.rotation().wxyz


def main():
    """Main function for YAM IK visualization."""
    viz = YamPyroki(rate=100.0, bimanual=True)
    viz.run()


if __name__ == "__main__":
    main()
