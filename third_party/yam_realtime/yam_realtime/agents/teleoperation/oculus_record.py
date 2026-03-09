# This agent is for the Oculus Quest 2 controller.
# It is used to teleoperate the robot using the Oculus Quest 2 controller.
# VR Controller is connected to Viser, which the is used to control the robot. 
import sys
sys.path.append('/home/sean/Desktop/YAM')
sys.path.append('/home/sean/Desktop/YAM/yam_realtime')

import threading
import time
from copy import deepcopy
from typing import Any, Dict, Optional

import numpy as np
import viser
import viser.extras
from dm_env.specs import Array

from yam_realtime.agents.agent import Agent
from yam_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
from yam_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad
from yam_realtime.utils.portal_utils import remote

from oculus_reader.oculus_reader.reader import OculusReader
import viser.transforms as vtf
from scipy.spatial.transform import Rotation as R

# left arm matrix mapping from viser to vr
vr_map_viser_left = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])

# right arm matrix mapping from viser to vr
vr_map_viser_right = np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
])

def vr_to_viser_pose(pose):
    return np.array([pose[2], pose[0], pose[1]])

def vr_to_viser_rot(rot_matrix, vr_map_viser):
    R_vr = rot_matrix
    Rm = vr_map_viser @ R_vr @ vr_map_viser.T

    # Re-orthonormalize (polar decomposition via SVD) and enforce det=+1
    U, S, Vt = np.linalg.svd(Rm)
    Rcorr = U @ Vt
    if np.linalg.det(Rcorr) < 0:
        U[:, 2] *= -1           # flip one axis to restore right-handedness
        Rcorr = U @ Vt
    return Rcorr

def rmat_to_quat(rot_mat):
    quat = R.from_matrix(rot_mat).as_quat()
    return quat

def euler_to_quat(euler):
    quat = R.from_euler("xyz", euler).as_quat()
    return quat

def quat_diff(quat1, quat2):
    quat1 = R.from_quat(quat1)
    quat2 = R.from_quat(quat2)
    quat_diff = quat1 * quat2.inv()
    return  quat_diff.as_quat()

class OculusRecordAgent(Agent):
    def __init__(self, bimanual: bool = False, right_arm_extrinsic: Optional[Dict[str, Any]] = None,
                 pos_scale: float = 0.5, rot_scale: float = 0.5, max_delta_pos: float = 0.02, max_delta_rot: float = 0.1, max_joint_delta: float = 0.05):
        self.right_arm_extrinsic = right_arm_extrinsic
        self.bimanual = bimanual
        self.pos_scale = pos_scale
        self.rot_scale = rot_scale
        self.max_delta_pos = max_delta_pos  # Max position delta per frame (meters)
        self.max_delta_rot = max_delta_rot  # Max rotation delta per frame (radians)
        self.max_joint_delta = max_joint_delta  # Max joint change per frame (rad)
        if bimanual:
            assert right_arm_extrinsic is not None, "right_arm_extrinsic must be provided for bimanual robot"
        self.reset_state()
        self.viser_server = viser.ViserServer()
        self.ik = YamPyroki(viser_server=self.viser_server, bimanual=bimanual)
        self.ik_thread = threading.Thread(target=self.ik.run)
        self.ik_thread.start()
        self.obs = None
        self.real_vis_thread = threading.Thread(target=self._update_visualization)
        self.real_vis_thread.start()
        self._setup_visualization()
        self.oculus_reader = OculusReader()
        self.oculus_reader_thread = threading.Thread(target=self._update_internal_state)
        self.oculus_reader_thread.start()
    
    def reset_state(self):
        self.state = {
            "poses" : {},
            "buttons" : {},
            "movement_enabled" : {"left": False, "right": False},
            "controller_on" : False,
        }
        self.update_VR = {"left": True, "right": True}   # update vr origin and state
        self.reset_origin = {"left": True, "right": True} # position
        self.viser_origin = {"left": None, "right": None}
        self.vr_origin = {"left": None, "right": None}
        self.vr_state = {"left": None, "right": None}
        self.vr_prev_state = {"left": None, "right": None}
        self.prev_joints = {"left": None, "right": None}

    def _update_internal_state(self, num_wait_sec = 5, hz=90):
        """Update VR state at high frequency (90Hz) to ensure smooth tracking.
        This should be faster than the control loop (30Hz) to avoid stale data."""
        last_read_time = time.time()
        while True:
            time.sleep(1/hz)

            time_since_last_read = time.time() - last_read_time

            poses, buttons = self.oculus_reader.get_transformations_and_buttons()
            # print(poses)
            # print(buttons)
            self.state["controller_on"] = time_since_last_read < num_wait_sec

            # if we didn't read anything, VR is off and we try again. 
            # if the time to get poses is too long, we set the controller_on to off.
            if poses == {}:
                continue

            # Determine if recentering is needed (if gripper is hold and then released)
            toggled = {"left": self.state["movement_enabled"]["left"] != buttons["LG"], "right": self.state["movement_enabled"]["right"] != buttons["RG"]}
            self.update_VR = {"left": self.update_VR["left"] or buttons["LG"], "right": self.update_VR["right"] or buttons["RG"]}
            self.reset_origin = {"left": self.reset_origin["left"] or toggled["left"], "right": self.reset_origin["right"] or toggled["right"]}
            # self.reset_orientation = self.reset_orientation or buttons["LJ"]
            # self.reset_orientation = self.reset_orientation or buttons["LJ"]

        
            # save readings from VR Controller
            self.state["movement_enabled"] = {"left": buttons["LG"], "right": buttons["RG"]}
            self.state["buttons"] = buttons
            self.state["poses"] = poses
            self.state["controller_on"] = True

            last_read_time = time.time()
        
    # Realign the axis of the VR controller to the robot base frame
    def _process_VR_reading_left(self):
        rot_mat_left = np.asarray(self.state["poses"]["l"])
        vr_pos_left = vr_to_viser_pose(rot_mat_left[:3, 3])
        vr_quat_left = rmat_to_quat(vr_to_viser_rot(rot_mat_left[:3, :3], vr_map_viser_left))
        vr_gripper_left = self.state["buttons"]["leftTrig"][0]

        self.vr_state["left"] = {"pose" : vr_pos_left, "quat" : vr_quat_left, "gripper": vr_gripper_left}

    def _process_VR_reading_right(self):
        rot_mat_right = np.asarray(self.state["poses"]["r"])
        vr_pos_right = vr_to_viser_pose(rot_mat_right[:3, 3])
        vr_quat_right = rmat_to_quat(vr_to_viser_rot(rot_mat_right[:3, :3], vr_map_viser_right))
        vr_gripper_right = self.state["buttons"]["rightTrig"][0]

        self.vr_state["right"] = {"pose" : vr_pos_right, "quat" : vr_quat_right, "gripper": vr_gripper_right}

    def _calc_delta_action(self, viser_state_dict):
        if self.state["poses"] == {}:
            return

        # Update VR state every frame when movement is enabled (for smooth tracking)
        # update_VR flag is only used to control origin reset, not VR reading
        if self.state["movement_enabled"]["left"]:
            self._process_VR_reading_left()
            if self.update_VR["left"]:
                self.update_VR["left"] = False

        
        if self.state["movement_enabled"]["right"]:
            self._process_VR_reading_right()
            if self.update_VR["right"]:
                self.update_VR["right"] = False
            

        # Read robot current state from the obs (Dict[str, Any])
        viser_pos_left = viser_state_dict["pos"]["left"][:3]
        viser_quat_left = viser_state_dict["pos"]["left"][3:]
        viser_gripper_left = viser_state_dict["gripper"]["left"]
        viser_pos_right = viser_state_dict["pos"]["right"][:3]
        viser_quat_right = viser_state_dict["pos"]["right"][3:]
        viser_gripper_right = viser_state_dict["gripper"]["right"]
        # print(robot_pos, robot_quat)

        # Reset origin if needed
        if self.reset_origin["left"] and self.vr_state["left"] is not None:
            self.viser_origin["left"] = {"pos": viser_pos_left, "quat": viser_quat_left, "gripper": viser_gripper_left}
            self.vr_origin["left"] = {"pos": self.vr_state["left"]["pose"].copy(), "quat": self.vr_state["left"]["quat"].copy(), "gripper": self.vr_state["left"]["gripper"]}
            self.reset_origin["left"] = False
            # Deep copy to avoid reference issues
            self.vr_prev_state["left"] = {"pose": self.vr_state["left"]["pose"].copy(), 
                                          "quat": self.vr_state["left"]["quat"].copy(), 
                                          "gripper": self.vr_state["left"]["gripper"]}
            
        if self.reset_origin["right"] and self.vr_state["right"] is not None:
            self.viser_origin["right"] = {"pos": viser_pos_right, "quat": viser_quat_right, "gripper": viser_gripper_right}
            self.vr_origin["right"] = {"pos": self.vr_state["right"]["pose"].copy(), "quat": self.vr_state["right"]["quat"].copy(), "gripper": self.vr_state["right"]["gripper"]}
            self.reset_origin["right"] = False
            # Deep copy to avoid reference issues
            self.vr_prev_state["right"] = {"pose": self.vr_state["right"]["pose"].copy(), 
                                           "quat": self.vr_state["right"]["quat"].copy(), 
                                           "gripper": self.vr_state["right"]["gripper"]}

        # Calculate delta action ande target action
        viser_desired_pos_left = viser_pos_left
        viser_desired_quat_left = viser_quat_left
        viser_desired_pos_right = viser_pos_right
        viser_desired_quat_right = viser_quat_right
        delta_pos_left = np.zeros(3)
        delta_quat_left = np.array([0.0, 0.0, 0.0, 1.0])
        delta_pos_right = np.zeros(3)
        delta_quat_right = np.array([0.0, 0.0, 0.0, 1.0])
        
        # Only calculate delta if we have both current and previous VR states AND movement is enabled
        if self.vr_state["left"] is not None and self.vr_prev_state["left"] is not None and self.state["movement_enabled"]["left"]:
            delta_pos_left = (self.vr_state["left"]["pose"] - self.vr_prev_state["left"]["pose"]) * self.pos_scale
            delta_quat_left = quat_diff(self.vr_state["left"]["quat"], self.vr_prev_state["left"]["quat"])   
            
            # Limit delta position magnitude
            delta_pos_norm = np.linalg.norm(delta_pos_left)
            if delta_pos_norm > self.max_delta_pos:
                delta_pos_left = delta_pos_left * (self.max_delta_pos / delta_pos_norm)
            
            # Apply rotation scaling and limit delta rotation magnitude
            delta_rot = R.from_quat(delta_quat_left)
            delta_rotvec = delta_rot.as_rotvec() * self.rot_scale
            delta_rot_norm = np.linalg.norm(delta_rotvec)
            if delta_rot_norm > self.max_delta_rot:
                delta_rotvec = delta_rotvec * (self.max_delta_rot / delta_rot_norm)
            delta_quat_left = R.from_rotvec(delta_rotvec).as_quat()
            
            viser_desired_pos_left = viser_pos_left + delta_pos_left
            viser_desired_quat_left = (R.from_quat(delta_quat_left) * R.from_quat(viser_quat_left)).as_quat()
        
        # Only calculate delta if we have both current and previous VR states AND movement is enabled
        if self.vr_state["right"] is not None and self.vr_prev_state["right"] is not None and self.state["movement_enabled"]["right"]:
            delta_pos_right = (self.vr_state["right"]["pose"] - self.vr_prev_state["right"]["pose"]) * self.pos_scale
            delta_quat_right = quat_diff(self.vr_state["right"]["quat"], self.vr_prev_state["right"]["quat"])   
            
            # Limit delta position magnitude
            delta_pos_norm = np.linalg.norm(delta_pos_right)
            if delta_pos_norm > self.max_delta_pos:
                delta_pos_right = delta_pos_right * (self.max_delta_pos / delta_pos_norm)
            
            # Apply rotation scaling and limit delta rotation magnitude
            delta_rot = R.from_quat(delta_quat_right)
            delta_rotvec = delta_rot.as_rotvec() * self.rot_scale
            delta_rot_norm = np.linalg.norm(delta_rotvec)
            if delta_rot_norm > self.max_delta_rot:
                delta_rotvec = delta_rotvec * (self.max_delta_rot / delta_rot_norm)
            delta_quat_right = R.from_rotvec(delta_rotvec).as_quat()
            
            viser_desired_pos_right = viser_pos_right + delta_pos_right
            viser_desired_quat_right = (R.from_quat(delta_quat_right) * R.from_quat(viser_quat_right)).as_quat()

        # Update prev state only when movement is enabled to avoid drift
        # Use deep copy to avoid reference issues between vr_state and vr_prev_state
        if self.state["movement_enabled"]["left"] and self.vr_state["left"] is not None:
            self.vr_prev_state["left"] = {"pose": self.vr_state["left"]["pose"].copy(), 
                                          "quat": self.vr_state["left"]["quat"].copy(), 
                                          "gripper": self.vr_state["left"]["gripper"]}
        if self.state["movement_enabled"]["right"] and self.vr_state["right"] is not None:
            self.vr_prev_state["right"] = {"pose": self.vr_state["right"]["pose"].copy(), 
                                           "quat": self.vr_state["right"]["quat"].copy(), 
                                           "gripper": self.vr_state["right"]["gripper"]}
        
        # Return the desired action
        return [np.concatenate([viser_desired_pos_left, [viser_desired_quat_left[3]], [viser_desired_quat_left[0]], [viser_desired_quat_left[1]], [viser_desired_quat_left[2]]]), 
        np.concatenate([viser_desired_pos_right, [viser_desired_quat_right[3]], [viser_desired_quat_right[0]], [viser_desired_quat_right[1]], [viser_desired_quat_right[2]]]),
        np.concatenate([delta_pos_left, [delta_quat_left[3]], delta_quat_left[:3]]),
        np.concatenate([delta_pos_right, [delta_quat_right[3]], delta_quat_right[:3]])]

    def _setup_visualization(self):
        self.base_frame_left_real = self.viser_server.scene.add_frame("/base_left_real", show_axes=False)
        self.urdf_vis_left_real = viser.extras.ViserUrdf(
            self.viser_server,
            self.ik.urdf,
            root_node_name="/base_left_real",
            mesh_color_override=(0.8, 0.5, 0.5),
        )
        for mesh in self.urdf_vis_left_real._meshes:
            mesh.opacity = 0.25  # type: ignore
        self.left_gripper_slider_handle = self.viser_server.gui.add_slider(
            "Right Gripper", min=0.0, max=1.0, step=0.01, initial_value=1.0
        )

        if self.bimanual and self.right_arm_extrinsic is not None:
            self.ik.base_frame_right.position = np.array(self.right_arm_extrinsic["position"])
            self.ik.base_frame_right.wxyz = np.array(self.right_arm_extrinsic["rotation"])
            self.base_frame_right_real = self.viser_server.scene.add_frame(
                "/base_left_real/base_right_real", show_axes=False
            )
            self.base_frame_right_real.position = self.ik.base_frame_right.position
            self.urdf_vis_right_real = viser.extras.ViserUrdf(
                self.viser_server,
                self.ik.urdf,
                root_node_name="/base_left_real/base_right_real",
                mesh_color_override=(0.8, 0.5, 0.5),
            )
            for mesh in self.urdf_vis_right_real._meshes:
                mesh.opacity = 0.25  # type: ignore
            self.right_gripper_slider_handle = self.viser_server.gui.add_slider(
                "Left Gripper", min=0.0, max=1.0, step=0.01, initial_value=1.0
            )

        self.viser_cam_img_handles = {}

    def _update_visualization(self):
        while self.obs is None:
            time.sleep(0.025)
        while True:
            if self.bimanual:
                self.urdf_vis_right_real.update_cfg(np.flip(self.obs["right"]["joint_pos"]))
            self.urdf_vis_left_real.update_cfg(np.flip(self.obs["left"]["joint_pos"]))

            # Extract RGB images from observation (if any)
            rgb_images = obs_get_rgb(self.obs)
            if rgb_images:
                for key in rgb_images.keys():
                    if key not in self.viser_cam_img_handles.keys():
                        self.viser_cam_img_handles[key] = self.viser_server.gui.add_image(rgb_images[key], label=key)
                    # resize viser images to 224x224
                    self.viser_cam_img_handles[key].image = resize_with_pad(rgb_images[key], 224, 224)

            time.sleep(0.02)

    def act(self, obs: Dict[str, Any]) -> Any:
        # if obs is empty, set the ik to the initial position. Use this as a trick so we can use move_joint to move the robot to the initial position.
        # move_joint doesn't advance the viser, so we need to set it ourselves. 
        if obs == {}:
            self.ik.transform_handles["left"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["left"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            self.ik.transform_handles["right"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["right"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            # Clear prev_joints on reset to prevent jump in next episode
            self.prev_joints = {"left": None, "right": None}
            # Reset VR origins to prevent cumulative drift between episodes
            self.reset_origin = {"left": True, "right": True}
            # Clear vr_prev_state so first frame won't have delta from old episode
            self.vr_prev_state = {"left": None, "right": None}
            return {}

        pose_left = np.asarray(self.ik.get_target_poses()["left"].translation())
        quat_left = np.asarray(self.ik.get_target_poses()["left"].rotation().wxyz)
        xyzw_left = np.concatenate([quat_left[1:], [quat_left[0]]])

        pose_right = np.asarray(self.ik.get_target_poses()["right"].translation())
        quat_right = np.asarray(self.ik.get_target_poses()["right"].rotation().wxyz)
        xyzw_right = np.concatenate([quat_right[1:], [quat_right[0]]])
        
        viser_state = {
            "pos": {"left": np.concatenate([pose_left, xyzw_left]), "right": np.concatenate([pose_right, xyzw_right])},
            "gripper" : {
                "left": obs["left"]["gripper_pos"] if "left" in obs else 0.0,
                "right": obs["right"]["gripper_pos"] if "right" in obs else 0.0
            },
        }

        viser_action = self._calc_delta_action(viser_state)
        if viser_action is not None:
            self.ik.transform_handles["left"].control.position = viser_action[0][:3]
            self.ik.transform_handles["left"].control.wxyz = viser_action[0][3:]
            # Only update gripper when movement is enabled (grip button pressed)
            if self.state["movement_enabled"]["left"]:
                self.left_gripper_slider_handle.value = 1 - self.state["buttons"].get("leftTrig", (0.0,))[0]
            self.ik.transform_handles["right"].control.position = viser_action[1][:3]
            self.ik.transform_handles["right"].control.wxyz = viser_action[1][3:]
            if self.state["movement_enabled"]["right"]:
                self.right_gripper_slider_handle.value = 1 - self.state["buttons"].get("rightTrig", (0.0,))[0]
            
            # Immediately solve IK to get synchronized joint values
            # This avoids relying on background IK thread timing
            self.ik.solve_ik()
        else:
            self.ik.transform_handles["left"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["left"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            self.ik.transform_handles["right"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["right"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])

        # Apply joint velocity limit to prevent IK jumps (after IK calculation)
        if self.prev_joints["left"] is not None:
            joint_diff = self.ik.joints["left"] - self.prev_joints["left"]
            joint_diff_clipped = np.clip(joint_diff, -self.max_joint_delta, self.max_joint_delta)
            if not np.allclose(joint_diff, joint_diff_clipped):
                self.ik.joints["left"] = self.prev_joints["left"] + joint_diff_clipped
        self.prev_joints["left"] = self.ik.joints["left"].copy()

        if self.bimanual and self.prev_joints["right"] is not None:
            joint_diff = self.ik.joints["right"] - self.prev_joints["right"]
            joint_diff_clipped = np.clip(joint_diff, -self.max_joint_delta, self.max_joint_delta)
            if not np.allclose(joint_diff, joint_diff_clipped):
                self.ik.joints["right"] = self.prev_joints["right"] + joint_diff_clipped
        if self.bimanual:
            self.prev_joints["right"] = self.ik.joints["right"].copy()

        action = {
            "left": {
                "pos": np.concatenate([np.flip(self.ik.joints["left"]), [self.left_gripper_slider_handle.value]]),
                "delta": viser_action[2]
            }
        }
        if self.bimanual:
            assert self.ik.joints.keys() == {"left", "right"}, (
                "bimanual mode must have both left and right joint ik solved"
            )
            action["right"] = {
                "pos": np.concatenate([np.flip(self.ik.joints["right"]), [self.right_gripper_slider_handle.value]]),
                "delta": viser_action[3]
            }

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Define the action specification."""
        action_spec = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32), "delta": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            action_spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32), "delta": Array(shape=(7,), dtype=np.float32)}
        return action_spec

    ##################################################################### function for collecting data ###################################################################

    @remote(serialization_needed=True)
    def get_info(self):
        try:
            # Support both left (X/Y) and right (A/B) controller buttons
            success = self.state["buttons"].get("X", False) or self.state["buttons"].get("A", False)
            failure = self.state["buttons"].get("Y", False) or self.state["buttons"].get("B", False)
            
            state_info = {
                "success" : success,
                "failure" : failure,
                "X" : self.state["buttons"]["X"],
                "Y" : self.state["buttons"]["Y"],
                "left_gripper" : self.state["buttons"].get("leftTrig", (0.0,))[0],
                "right_gripper" : self.state["buttons"].get("rightTrig", (0.0,))[0],
                "movement_enabled" : self.state["movement_enabled"],
                "controller_on" : self.state["controller_on"],
            }
        except Exception as e:
            raise KeyError(
                "VR controllers are not connected!!!"
            )
        return state_info
