# type: ignore
import collections
from pathlib import Path
from typing import Optional, Union

import numpy as np
from dm_env.specs import Array

from yam_realtime.agents.agent import PolicyAgent
from yam_realtime.agents.constants import ActionSpec
from yam_realtime.utils.portal_utils import remote
from yam_realtime.robots.inverse_kinematics.yam_pyroki import YamPyroki
import threading
import time
from typing import Any, Dict, Optional
import viser
import viser.extras
from yam_realtime.sensors.cameras.camera_utils import obs_get_rgb, resize_with_pad
from scipy.spatial.transform import Rotation as R

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


class AsyncDiffusionAgent(PolicyAgent):
    def __init__(
        self,
        bimanual: bool = False,
        right_arm_extrinsic: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__()
        self.right_arm_extrinsic = right_arm_extrinsic
        self.bimanual = bimanual
        if bimanual:
            assert right_arm_extrinsic is not None, "right_arm_extrinsic must be provided for bimanual robot"
        self.viser_server = viser.ViserServer()
        self.ik = YamPyroki(viser_server=self.viser_server, bimanual=bimanual)
        self.ik_thread = threading.Thread(target=self.ik.run)
        self.ik_thread.start()
        self.obs = None
        self.real_vis_thread = threading.Thread(target=self._update_visualization)
        self.real_vis_thread.start()
        self._setup_visualization()

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
                "Right Gripper", min=0.0, max=2.4, step=0.01, initial_value=0.0
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
                    "Left Gripper", min=0.0, max=2.4, step=0.01, initial_value=0.0
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

    def _calc_delta_action(self, viser_state_dict):
        # Read robot current state from the obs (Dict[str, Any])
        viser_pos_left = viser_state_dict["pos"]["left"][:3]
        viser_quat_left = viser_state_dict["pos"]["left"][3:]

        viser_pos_right = viser_state_dict["pos"]["right"][:3]
        viser_quat_right = viser_state_dict["pos"]["right"][3:]

        delta_pos_left = viser_state_dict["delta_action"]["left"][:3]
        delta_quat_left = viser_state_dict["delta_action"]["left"][3:]
        delta_pos_right = viser_state_dict["delta_action"]["right"][:3]
        delta_quat_right = viser_state_dict["delta_action"]["right"][3:]
        # print(robot_pos, robot_quat)
        
        viser_desired_pos_left = viser_pos_left + delta_pos_left
        viser_desired_quat_left = (R.from_quat(delta_quat_left) * R.from_quat(viser_quat_left)).as_quat()
        

        viser_desired_pos_right = viser_pos_right + delta_pos_right
        viser_desired_quat_right = (R.from_quat(delta_quat_right) * R.from_quat(viser_quat_right)).as_quat()

        
        # Return the desired action
        return [np.concatenate([viser_desired_pos_left, [viser_desired_quat_left[3]], [viser_desired_quat_left[0]], [viser_desired_quat_left[1]], [viser_desired_quat_left[2]]]), 
        np.concatenate([viser_desired_pos_right, [viser_desired_quat_right[3]], [viser_desired_quat_right[0]], [viser_desired_quat_right[1]], [viser_desired_quat_right[2]]])]

    def act(self, obs: Dict[str, Any]) -> Any:
        if obs == {}:
            self.ik.transform_handles["left"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["left"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            self.ik.transform_handles["right"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["right"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            return {}
        pose_left = np.asarray(self.ik.get_target_poses()["left"].translation())
        quat_left = np.asarray(self.ik.get_target_poses()["left"].rotation().wxyz)
        xyzw_left = np.concatenate([quat_left[1:], [quat_left[0]]])

        pose_right = np.asarray(self.ik.get_target_poses()["right"].translation())
        quat_right = np.asarray(self.ik.get_target_poses()["right"].rotation().wxyz)
        xyzw_right = np.concatenate([quat_right[1:], [quat_right[0]]])
        
        viser_state = {
            "pos": {"left": np.concatenate([pose_left, xyzw_left]), "right": np.concatenate([pose_right, xyzw_right])}
        }

        if "delta_action" in obs:
            viser_state['delta_action'] = {'left': np.concatenate([obs['delta_action']['left'][:3], obs['delta_action']['left'][4:7], [obs['delta_action']['left'][3]]]), 
                                            'right': np.concatenate([obs['delta_action']['right'][:3], obs['delta_action']['right'][4:7], [obs['delta_action']['right'][3]]])}
        viser_action = self._calc_delta_action(viser_state)
        if viser_action is not None:
            self.ik.transform_handles["left"].control.position = viser_action[0][:3]
            self.ik.transform_handles["left"].control.wxyz = viser_action[0][3:]
            self.left_gripper_slider_handle.value = obs['delta_action']['left'][-1]
            self.ik.transform_handles["right"].control.position = viser_action[1][:3]
            self.ik.transform_handles["right"].control.wxyz = viser_action[1][3:]
            self.right_gripper_slider_handle.value = obs['delta_action']['right'][-1]
        else:
            self.ik.transform_handles["left"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["left"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            self.ik.transform_handles["right"].control.position = np.array([0.12, 0.00535176, 0.09107439])
            self.ik.transform_handles["right"].control.wxyz = np.array([0.5, 0.5, 0.5, 0.5])
            self.left_gripper_slider_handle.value = 1.0
            self.right_gripper_slider_handle.value = 1.0

        action = {
            "left": {
                "pos": np.concatenate([np.flip(self.ik.joints["left"]), [self.left_gripper_slider_handle.value]]),
            }
        }
        if self.bimanual:
            assert self.ik.joints.keys() == {"left", "right"}, (
                "bimanual mode must have both left and right joint ik solved"
            )
            action["right"] = {
                "pos": np.concatenate([np.flip(self.ik.joints["right"]), [self.right_gripper_slider_handle.value]]),
            }

        return action

    @remote(serialization_needed=True)
    def action_spec(self) -> Dict[str, Dict[str, Array]]:
        """Define the action specification."""
        action_spec = {
            "left": {"pos": Array(shape=(7,), dtype=np.float32)},
        }
        if self.bimanual:
            action_spec["right"] = {"pos": Array(shape=(7,), dtype=np.float32)}
        return action_spec
