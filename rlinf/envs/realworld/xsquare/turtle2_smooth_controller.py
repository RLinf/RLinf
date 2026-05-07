# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import threading
import time
import tracemalloc

import numpy as np
import rospy
from cv_bridge import CvBridge

# import rospkg
# rospack = rospkg.RosPack()
# package_path = rospack.get_path('turtle2_controller')
# sys.path.append(os.path.join(rospack_path, 'turtle2_controller'))
from turtle2_basic.turtle2_controller.Turtle2Controller import Turtle2Controller

from rlinf.scheduler import Cluster, NodePlacementStrategy, Worker
from rlinf.utils.logging import get_logger

from .turtle2_robot_state import Turtle2RobotState


class Turtle2SmoothController(Worker):
    """Controller for turtle2 robot, XSquare"""

    @staticmethod
    def launch_controller(
        freq: int = 50,
        env_idx: int = 0,
        node_rank: int = 0,
        worker_rank: int = 0,
        debug_pose_control: bool = False,
        debug_gripper_control: bool = False,
        gripper_target_tolerance: float = 0.05,
        pose_control_backend: str = "smooth",
        takeover_publish_hz: float = 100.0,
        follower_pose_cmd_left_topic: str = "/follow_pos_cmd_1",
        follower_pose_cmd_right_topic: str = "/follow_pos_cmd_2",
    ):
        """Launch a Turtle2SmoothController on the specified worker's node.

        Args:
            freq (int): The interpolate frequency for the controller.
            node_rank (int): The rank of the node to launch the controller on.
            worker_rank (int): The rank of the env worker to the controller is associated with.

        Returns:
            Turtle2SmoothController: The launched Turtle2SmoothController instance.
        """
        cluster = Cluster()
        placement = NodePlacementStrategy(node_ranks=[node_rank])
        return Turtle2SmoothController.create_group(
            freq,
            debug_pose_control,
            debug_gripper_control,
            gripper_target_tolerance,
            pose_control_backend,
            takeover_publish_hz,
            follower_pose_cmd_left_topic,
            follower_pose_cmd_right_topic,
        ).launch(
            cluster=cluster,
            placement_strategy=placement,
            name=f"Turtle2SmoothController-{worker_rank}-{env_idx}",
        )

    def __init__(
        self,
        freq=50,
        debug_pose_control: bool = False,
        debug_gripper_control: bool = False,
        gripper_target_tolerance: float = 0.05,
        pose_control_backend: str = "smooth",
        takeover_publish_hz: float = 100.0,
        follower_pose_cmd_left_topic: str = "/follow_pos_cmd_1",
        follower_pose_cmd_right_topic: str = "/follow_pos_cmd_2",
    ):
        super().__init__()
        self._logger = get_logger()
        # FIXME: should move to roscontroller
        rospy.init_node("Turtle2_Smooth_Controller_Node")
        self.bridge = CvBridge()
        # FIXME: should rewrite with roscontroller
        self.controller = Turtle2Controller()

        self.controller.chassis_set_current_pose_as_virtual_zero()

        self._state = Turtle2RobotState()
        self.pose_control_backend = str(pose_control_backend).lower()
        if self.pose_control_backend not in {"smooth", "hybrid"}:
            raise ValueError(
                "pose_control_backend must be one of {'smooth', 'hybrid'}, "
                f"got {self.pose_control_backend!r}."
            )
        self.takeover_publish_hz = float(takeover_publish_hz)
        if self.takeover_publish_hz <= 0:
            raise ValueError("takeover_publish_hz must be positive.")
        self._follower_pose_cmd_left_topic = follower_pose_cmd_left_topic
        self._follower_pose_cmd_right_topic = follower_pose_cmd_right_topic
        self._direct_pose_msg_cls = None
        self._direct_pose_left_pub = None
        self._direct_pose_right_pub = None
        self._direct_pose_left_target = None
        self._direct_pose_right_target = None
        self._takeover_gate = threading.Event()
        self._direct_target_lock = threading.RLock()

        control_period = rospy.Duration(1 / freq)
        state_period = rospy.Duration(1 / 200.0)

        self.left_arm_target = [0, 0, 0, 0, 0, 0, 0]
        self.right_arm_target = [0, 0, 0, 0, 0, 0, 0]

        self.last_expected_xyz1 = None
        self.last_expected_xyz2 = None
        self.last_expected_rpy1 = None
        self.last_expected_rpy2 = None

        # xyz, rpy, gripper
        self.tol = [0.002, 0.005, 5]  # m, rad, cm
        self.gripper_target_tolerance = float(gripper_target_tolerance)
        self.debug_pose_control = bool(debug_pose_control)
        self.debug_gripper_control = bool(debug_gripper_control)
        self.xyz_speed = 0.5  # m/s
        self.rpy_speed = 1.5  # rad/s
        self.freq = freq

        # FIXME: should move to roscontroller
        rospy.Timer(control_period, self.smooth_action_callback)
        direct_period = rospy.Duration(1 / self.takeover_publish_hz)
        rospy.Timer(direct_period, self.direct_pose_callback)
        rospy.Timer(state_period, self.state_callback)

        tracemalloc.start(15)
        self.snapshot_base = tracemalloc.take_snapshot()

    def _update_state_from_controller(self):
        arms_data = self.controller.arms_data()
        self._state.follow1_pos = np.array(arms_data[0], dtype=np.float32)
        self._state.follow2_pos = np.array(arms_data[1], dtype=np.float32)
        joint_data = self.controller.arms_joint_data()
        self._state.follow1_joints = np.array(joint_data[0], dtype=np.float32)
        self._state.follow2_joints = np.array(joint_data[1], dtype=np.float32)
        cur_data = self.controller.arms_cur_data()
        self._state.follow1_cur_data = np.array(cur_data[0], dtype=np.float32)
        self._state.follow2_cur_data = np.array(cur_data[1], dtype=np.float32)
        head_data = self.controller.head_data()
        self._state.head_pos = np.array(head_data, dtype=np.float32)
        self._state.lift = float(self.controller.lift_data())
        chassis_pose = self.controller.chassis_pose_data()
        self._state.car_pose = np.array(chassis_pose, dtype=np.float32)
        return self._state

    def state_callback(self, event):
        self._update_state_from_controller()

    def get_state(self):
        return self._state

    def smooth_action_callback(self, event):
        if self._takeover_gate.is_set():
            return
        # print("intimer")
        xyz_step = self.xyz_speed / self.freq  # m
        rpy_step = self.rpy_speed / self.freq  # rad
        # start_time = time.time()

        curxyz1 = self._state.follow1_pos[0:3]
        curxyz2 = self._state.follow2_pos[0:3]
        # print("current pos:")
        # print(curxyz1, curxyz2)
        targetxyz1 = np.array(self.left_arm_target[0:3], dtype=float)
        targetxyz2 = np.array(self.right_arm_target[0:3], dtype=float)
        # print("target pos:")
        # print(targetxyz1, targetxyz2)
        errxyz1 = np.linalg.norm(curxyz1 - targetxyz1)
        errxyz2 = np.linalg.norm(curxyz2 - targetxyz2)

        currpy1 = self._state.follow1_pos[3:6]
        currpy2 = self._state.follow2_pos[3:6]
        # print("current rpy:")
        # print(currpy1, currpy2)
        targetrpy1 = np.array(self.left_arm_target[3:6], dtype=float)
        targetrpy2 = np.array(self.right_arm_target[3:6], dtype=float)
        # print("target rpy:")
        # print(targetrpy1, targetrpy2)
        errrpy1 = np.linalg.norm(currpy1 - targetrpy1)
        errrpy2 = np.linalg.norm(currpy2 - targetrpy2)

        if (
            errxyz1 < self.tol[0]
            and errxyz2 < self.tol[0]
            and errrpy1 < self.tol[1]
            and errrpy2 < self.tol[1]
        ):
            # print(f"[INFO] target reach! {errxyz1:.4f}, {errxyz2:.4f}, {errrpy1:.4f}, {errrpy2:.4f}")
            self.last_expected_xyz1 = curxyz1.copy()
            self.last_expected_xyz2 = curxyz2.copy()
            self.last_expected_rpy1 = currpy1.copy()
            self.last_expected_rpy2 = currpy2.copy()
            return
        else:
            # interpolate xyz
            curxyz1 = (
                0.5 * (curxyz1 + self.last_expected_xyz1)
                if self.last_expected_xyz1 is not None
                else curxyz1
            )
            curxyz2 = (
                0.5 * (curxyz2 + self.last_expected_xyz2)
                if self.last_expected_xyz2 is not None
                else curxyz2
            )
            currpy1 = (
                0.5 * (currpy1 + self.last_expected_rpy1)
                if self.last_expected_rpy1 is not None
                else currpy1
            )
            currpy2 = (
                0.5 * (currpy2 + self.last_expected_rpy2)
                if self.last_expected_rpy2 is not None
                else currpy2
            )

            dirxyz1 = (targetxyz1 - curxyz1) / (errxyz1 + 0.001)
            dirxyz2 = (targetxyz2 - curxyz2) / (errxyz2 + 0.001)
            # print("dirxyz2:",dirxyz2)
            stepxyz1 = dirxyz1 * min(xyz_step, errxyz1)
            stepxyz2 = dirxyz2 * min(xyz_step, errxyz2)
            # print("stepxyz2:",stepxyz2)
            newxyz1 = curxyz1 + stepxyz1
            self.last_expected_xyz1 = newxyz1.copy()

            newxyz2 = curxyz2 + stepxyz2
            self.last_expected_xyz2 = newxyz2.copy()

            # interpolate rpy
            dirrpy1 = (targetrpy1 - currpy1) / (errrpy1 + 0.001)
            dirrpy2 = (targetrpy2 - currpy2) / (errrpy2 + 0.001)
            # print("dirrpy2:",dirrpy2)
            steprpy1 = dirrpy1 * min(rpy_step, errrpy1)
            steprpy2 = dirrpy2 * min(rpy_step, errrpy2)
            newrpy1 = currpy1 + steprpy1
            self.last_expected_rpy1 = newrpy1.copy()

            newrpy2 = currpy2 + steprpy2
            # print("last_exp:", self.last_expected_rpy2, "; stp:", steprpy2)
            self.last_expected_rpy2 = newrpy2.copy()

            newpos1 = [
                newxyz1[0],
                newxyz1[1],
                newxyz1[2],
                newrpy1[0],
                newrpy1[1],
                newrpy1[2],
                self.left_arm_target[6],
            ]
            newpos2 = [
                newxyz2[0],
                newxyz2[1],
                newxyz2[2],
                newrpy2[0],
                newrpy2[1],
                newrpy2[2],
                self.right_arm_target[6],
            ]
            # print("new pos:",newpos2)
            self.controller.arms_control(newpos1, newpos2)
            # time.sleep(0.2 / self.freq)

    def move_arm(self, left_arm_target, right_arm_target):
        self._validate_pose_target(left_arm_target, right_arm_target)
        self.left_arm_target = left_arm_target
        self.right_arm_target = right_arm_target

    def _validate_pose_target(self, left_arm_target, right_arm_target) -> None:
        assert isinstance(left_arm_target, list) and len(left_arm_target) == 7, (
            "left_arm_target should be a list of length 7"
        )
        assert isinstance(right_arm_target, list) and len(right_arm_target) == 7, (
            "right_arm_target should be a list of length 7"
        )
        assert np.all(np.isfinite(left_arm_target)), "left_arm_target must be finite."
        assert np.all(np.isfinite(right_arm_target)), "right_arm_target must be finite."

    def set_takeover_gate(self, enabled: bool) -> None:
        if enabled:
            self._takeover_gate.set()
        else:
            self._takeover_gate.clear()

    def is_takeover_gate_active(self) -> bool:
        return self._takeover_gate.is_set()

    def clear_takeover_target(self) -> None:
        with self._direct_target_lock:
            self._direct_pose_left_target = None
            self._direct_pose_right_target = None

    def publish_takeover_pose(self, left_arm_target, right_arm_target) -> None:
        self._validate_pose_target(left_arm_target, right_arm_target)
        with self._direct_target_lock:
            self._direct_pose_left_target = list(left_arm_target)
            self._direct_pose_right_target = list(right_arm_target)
            left_target = list(self._direct_pose_left_target)
            right_target = list(self._direct_pose_right_target)
        self._publish_direct_pose(left_target, right_target)

    def direct_pose_callback(self, event):
        if not self._takeover_gate.is_set():
            return
        with self._direct_target_lock:
            if (
                self._direct_pose_left_target is None
                or self._direct_pose_right_target is None
            ):
                return
            left_target = list(self._direct_pose_left_target)
            right_target = list(self._direct_pose_right_target)
        self._publish_direct_pose(left_target, right_target)

    def _ensure_direct_pose_publishers(self):
        if (
            self._direct_pose_msg_cls is not None
            and self._direct_pose_left_pub is not None
            and self._direct_pose_right_pub is not None
        ):
            return
        try:
            from arm_control.msg import PosCmd
        except ImportError:
            from communicationPort.msg import PosCmd

        self._direct_pose_msg_cls = PosCmd
        self._direct_pose_left_pub = rospy.Publisher(
            self._follower_pose_cmd_left_topic,
            PosCmd,
            queue_size=10,
        )
        self._direct_pose_right_pub = rospy.Publisher(
            self._follower_pose_cmd_right_topic,
            PosCmd,
            queue_size=10,
        )

    def _make_pos_cmd_msg(self, pose):
        msg = self._direct_pose_msg_cls()
        msg.x = float(pose[0])
        msg.y = float(pose[1])
        msg.z = float(pose[2])
        msg.roll = float(pose[3])
        msg.pitch = float(pose[4])
        msg.yaw = float(pose[5])
        msg.gripper = float(pose[6])
        if hasattr(msg, "mode1"):
            msg.mode1 = 0
        if hasattr(msg, "mode2"):
            msg.mode2 = 0
        return msg

    def _publish_direct_pose(self, left_pose, right_pose):
        self._ensure_direct_pose_publishers()
        self._direct_pose_left_pub.publish(self._make_pos_cmd_msg(left_pose))
        self._direct_pose_right_pub.publish(self._make_pos_cmd_msg(right_pose))

    def _sync_smooth_targets_from_state(self, state):
        left_pose = state.follow1_pos.astype(np.float32, copy=True)
        right_pose = state.follow2_pos.astype(np.float32, copy=True)
        self.left_arm_target = left_pose.tolist()
        self.right_arm_target = right_pose.tolist()
        self.last_expected_xyz1 = left_pose[:3].copy()
        self.last_expected_xyz2 = right_pose[:3].copy()
        self.last_expected_rpy1 = left_pose[3:6].copy()
        self.last_expected_rpy2 = right_pose[3:6].copy()
        return left_pose, right_pose

    def sync_smooth_target_to_current_pose(self):
        state = self._update_state_from_controller()
        self._sync_smooth_targets_from_state(state)
        return state

    def hold_current_pose(self):
        state = self._update_state_from_controller()
        left_pose, right_pose = self._sync_smooth_targets_from_state(state)

        if self._takeover_gate.is_set():
            self.publish_takeover_pose(left_pose.tolist(), right_pose.tolist())
        else:
            self.controller.arms_control(self.left_arm_target, self.right_arm_target)
        return state

    def reset_arms(self):
        self.left_arm_target = [0, 0, 0, 0, 0, 0, 0]
        self.right_arm_target = [0, 0, 0, 0, 0, 0, 0]
        self.log_info("Reset target to zero.")
        time.sleep(2.0)

    def check_cams(self, timeout=0.5):
        cam1_ok = self.controller.cam.check_cam1(timeout)
        cam2_ok = self.controller.cam.check_cam2(timeout)
        cam3_ok = self.controller.cam.check_cam3(timeout)
        return cam1_ok, cam2_ok, cam3_ok

    def get_cams(self, ids):
        assert len(ids) > 0 and len(ids) <= 3
        frames = []
        for cam_id in ids:
            if cam_id == 0:
                frame1 = self.controller.cam.get_cam1_data()
                frames.append(frame1)
            elif cam_id == 1:
                frame2 = self.controller.cam.get_cam2_data()
                frames.append(frame2)
            elif cam_id == 2:
                frame3 = self.controller.cam.get_cam3_data()
                frames.append(frame3)
        assert len(frames) == len(ids), "get frames failed."
        return frames
