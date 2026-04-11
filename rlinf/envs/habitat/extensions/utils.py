# Copyright 2025 The RLinf Authors.
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

# This file contains code copied from the VLN-CE project: https://github.com/jacobkrantz/VLN-CE.
# The original code is licensed under the MIT License.

import textwrap
from copy import deepcopy
from typing import Any, Optional, Union

import habitat_sim
import numpy as np
import quaternion
import torch
from habitat.core.simulator import Simulator

try:
    from habitat.core.utils import try_cv2_import

    cv2 = try_cv2_import()
except ImportError:
    import cv2
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
    quaternion_to_list,
)
from habitat.utils.visualizations import maps as habitat_maps

# from habitat.habitat_baselines.common.tensorboard_utils import TensorboardWriter
from numpy import ndarray
from torch import Tensor

from rlinf.envs.habitat.extensions import maps


def observations_to_image(
    observation: dict[str, Any],
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = {}
    observation_size = -1
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"][:, :, :3]
        egocentric_view["rgb"] = rgb

    # draw depth map if observation has depth info. resize to rgb size.
    if "depth" in observation:
        if observation_size == -1:
            observation_size = observation["depth"].shape[0]
        depth_map = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        depth_map = cv2.resize(
            depth_map,
            dsize=(observation_size, observation_size),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view["depth"] = depth_map

    assert len(egocentric_view) > 0, "Expected at least one visual sensor enabled."

    if info is None:
        return egocentric_view

    map_k = None
    if "top_down_map_vlnce" in info:
        map_k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        map_k = "top_down_map"

    if map_k is not None:
        td_map = info[map_k]["map"]

        td_map = maps.colorize_topdown_map(
            td_map,
            info[map_k]["fog_of_war_mask"],
            fog_of_war_desat_amount=0.75,
        )
        td_map = habitat_maps.draw_agent(
            image=td_map,
            agent_center_coord=info[map_k]["agent_map_coord"][0],
            agent_rotation=info[map_k]["agent_angle"][0],
            agent_radius_px=min(td_map.shape[0:2]) // 24,
        )
        if td_map.shape[1] < td_map.shape[0]:
            td_map = np.rot90(td_map, 1)

        if td_map.shape[0] > td_map.shape[1]:
            td_map = np.rot90(td_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = td_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        td_map = cv2.resize(
            td_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view["top_down_map"] = td_map

    return egocentric_view


def pano_observations_to_image(
    observation: dict[str, Any], info: dict[str, Any]
) -> ndarray:
    """Creates a rudimentary frame for a panoramic observation. Includes RGB,
    depth, and a top-down map.
    TODO: create a visually-pleasing stitched panorama frame
    """
    pano_frame = []
    channels = None
    rgb = None
    if "rgb" in observation:
        cnt = observation["rgb"].shape[0]
        rgb = observation["rgb"][[*range(cnt // 2, cnt), *range(cnt // 2)], :, :, :]
        channels = rgb.shape[3]
        vert_bar = np.ones((rgb.shape[1], 20, channels)) * 255
        rgb_frame = [rgb[0]]
        for i in range(1, rgb.shape[0]):
            rgb_frame.append(vert_bar)
            rgb_frame.append(rgb[i])
        pano_frame.append(np.concatenate(rgb_frame, axis=1))

    if "depth" in observation:
        cnt = observation["depth"].shape[0]
        observation["depth"] = observation["depth"][
            [*range(cnt // 2, cnt), *range(cnt // 2)], :, :, :
        ]
        if len(pano_frame) > 0:
            assert observation["depth"].shape[0] == rgb.shape[0]
            pano_frame.append(np.ones((20, pano_frame[0].shape[1], channels)) * 255)
            observation_size = rgb.shape[1:3]
        else:
            observation_size = observation["depth"].shape[1:3]

        vert_bar = np.ones((observation_size[0], 20, 3)) * 255

        depth = (observation["depth"].squeeze() * 255).astype(np.uint8)
        depth = np.stack([depth for _ in range(3)], axis=3)

        depth_frame = [
            cv2.resize(depth[0], dsize=observation_size, interpolation=cv2.INTER_CUBIC)
        ]
        for i in range(1, depth.shape[0]):
            depth_frame.append(vert_bar)
            depth_frame.append(
                cv2.resize(
                    depth[i],
                    dsize=observation_size,
                    interpolation=cv2.INTER_CUBIC,
                )
            )
        pano_frame.append(np.concatenate(depth_frame, axis=1))

    pano_frame = np.concatenate(pano_frame, axis=0)

    if "top_down_map_vlnce" in info:
        k = "top_down_map_vlnce"
    elif "top_down_map" in info:
        k = "top_down_map"
    else:
        k = None

    if k is not None:
        top_down_map = info[k]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info[k]["fog_of_war_mask"]
        )
        map_agent_pos = info[k]["agent_map_coord"]
        top_down_map = habitat_maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info[k]["agent_angle"],
            agent_radius_px=min(top_down_map.shape[0:2]) // 24,
        )
        if top_down_map.shape[1] < top_down_map.shape[0]:
            top_down_map = np.rot90(top_down_map, 1)

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_width = pano_frame.shape[1] // 3
        top_down_height = int(top_down_width / old_w * old_h)

        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        white = (
            np.ones((top_down_height, pano_frame.shape[1] - top_down_width, 3)) * 255
        )
        top_down_map = np.concatenate((white, top_down_map), axis=1)
        pano_frame = np.concatenate((pano_frame, top_down_map), axis=0)

    return pano_frame.astype(np.uint8)


def add_id_on_img(img: ndarray, txt_id: str) -> ndarray:
    img_height = img.shape[0]
    img_width = img.shape[1]
    white = np.ones((10, img.shape[1], 3)) * 255
    img = np.concatenate((img, white), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    thickness = 2
    text_width = cv2.getTextSize(txt_id, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        txt_id,
        (start_width, img_height),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def add_instruction_on_img(img: ndarray, text: str) -> None:
    font_size = 1.1
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int((img.shape[1] - 15) / char_size[0]))
    if len(wrapped_text) < 8:
        wrapped_text.insert(0, "")

    y = 0
    start_x = 15
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 25
        cv2.putText(
            img,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )


def add_step_stats_on_img(
    img: ndarray,
    offset: Optional[float] = None,
    offset_mode: Optional[float] = None,
    distance: Optional[float] = None,
    distance_mode: Optional[float] = None,
    append_above: bool = True,
) -> ndarray:
    h, w, c = img.shape
    font_size = 0.9
    thickness = 2
    font = cv2.FONT_HERSHEY_SIMPLEX
    blank_image = (np.zeros(img.shape, dtype=np.uint8) + 1) * 255

    text = ""
    if offset is not None:
        if offset_mode is not None:
            text += f"  ofst/mode: {offset}/{offset_mode}"
        else:
            text += f"  ofst: {offset}"
    if distance is not None:
        if distance_mode is not None:
            text += f"  dist/mode: {distance}/{distance_mode}"
        else:
            text += f"  dist: {distance}"
    text = text.lstrip()
    if len(text) == 0:
        return img

    char_size = cv2.getTextSize(" ", font, font_size, thickness)[0]
    wrapped_text = textwrap.wrap(text, width=int(w / char_size[0]))

    y = 0
    max_width_to_center = max(
        int(cv2.getTextSize(wt, font, font_size, thickness)[0][0] / 2)
        for wt in wrapped_text
    )
    start_x = int(img.shape[1] / 2) - max_width_to_center
    for line in wrapped_text:
        textsize = cv2.getTextSize(line, font, font_size, thickness)[0]
        y += textsize[1] + 40
        cv2.putText(
            blank_image,
            line,
            (start_x, y),
            font,
            font_size,
            (0, 0, 0),
            thickness,
            lineType=cv2.LINE_AA,
        )

    text_image = blank_image[0 : y + 20, 0:w]
    top = np.ones([30, img.shape[1], img.shape[2]], dtype=np.uint8) * 255
    text_image = np.concatenate((top, text_image), axis=0)

    img = (text_image, img) if append_above else (img, text_image)
    return np.concatenate(img, axis=0)


def add_prob_on_img(img: ndarray, probability: float, pano_selected: bool) -> ndarray:
    img_height = img.shape[0]
    img_width = img.shape[1]
    white = np.ones((20, img.shape[1], 3)) * 255
    img = np.concatenate((img, white), axis=0)

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.7
    thickness = 2 if pano_selected else 1
    text_width = cv2.getTextSize(probability, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        probability,
        (start_width, img_height + 10),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def add_stop_prob_on_img(img: ndarray, stop: float, selected: bool) -> ndarray:
    img_width = img.shape[1]
    txt = "stop: " + str(round(stop, 2))

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.5
    thickness = 2 if selected else 1
    text_width = cv2.getTextSize(txt, font, font_size, thickness)[0][0]
    start_width = int(img_width / 2 - text_width / 2)
    cv2.putText(
        img,
        txt,
        (start_width, 20),
        font,
        font_size,
        (0, 0, 0),
        thickness,
        lineType=cv2.LINE_AA,
    )
    return img


def waypoint_observations_to_image(
    observation: dict[str, Any],
    info: dict[str, Any],
    pano_distribution: ndarray = None,
    agent_action_elements: Optional[dict[str, float]] = None,
    agent_stop: bool = False,
    distribution_modes: Optional[dict[str, float]] = None,
    predict_offset: bool = False,
    predict_distance: bool = False,
    agent_position: Optional[Tensor] = None,
    agent_heading: Optional[float] = None,
    oracle_action_elements: Optional[dict[str, float]] = None,
    oracle_stop: bool = False,
    num_panos: int = 12,
) -> ndarray:
    """Generates an image frame that combines an instruction, RGB observation,
    top down map, and waypoint variables.
    """

    def preds_to_coords(p, o, d):
        return predictions_to_global_coordinates(
            p, o, d, agent_position, agent_heading, num_panos
        )

    offset = None
    offset_mode = None
    distance = None
    distance_mode = None
    oracle_waypoint = None
    waypoint = None

    if agent_action_elements is not None:
        p = agent_action_elements["pano"]
        o = agent_action_elements["offset"]
        d = agent_action_elements["distance"]
        if not agent_stop:
            waypoint = preds_to_coords(p, o, d).squeeze(0)
        if predict_offset:
            offset = round(o, 2)
            if distribution_modes is not None:
                offset_mode = round(distribution_modes["offset"], 2)
        if predict_distance:
            distance = round(d, 2)
            if distribution_modes is not None:
                distance_mode = round(distribution_modes["distance"], 2)

    if not oracle_stop and oracle_action_elements is not None:
        oracle_waypoint = preds_to_coords(
            oracle_action_elements["pano"],
            oracle_action_elements["offset"],
            oracle_action_elements["distance"],
        ).squeeze(0)

    frame = None
    frame_width = 2048
    if "rgb" in observation:
        rgb = [
            add_id_on_img(
                observation["rgb"][i][:, 80 : (observation["rgb"][i].shape[1] - 80), :],
                str(i),
            )
            for i in range(observation["rgb"].shape[0])
        ]
        rgb = [
            add_prob_on_img(f, str(round(p, 2)), i == agent_action_elements["pano"])
            for i, (f, p) in enumerate(zip(rgb, pano_distribution[:-1].tolist()))
        ][::-1]
        rgb = rgb[6:] + rgb[:6]
        vertical_bar = np.ones((rgb[0].shape[0], 1, 3)) * 255
        for i in list(reversed(range(len(rgb) + 1))):
            rgb.insert(i, vertical_bar)

        rgb = np.concatenate(rgb, axis=1).astype(np.uint8)
        horizontal_bar = np.ones((10, rgb.shape[1], 3)) * 255

        stop_prob = add_stop_prob_on_img(
            np.ones((40, rgb.shape[1], rgb.shape[2])) * 255,
            pano_distribution[-1],
            pano_distribution.shape[0] == agent_action_elements["pano"] + 1,
        )
        rgb = np.concatenate(
            [horizontal_bar, rgb, stop_prob, horizontal_bar], axis=0
        ).astype(np.uint8)
        new_height = int((frame_width / rgb.shape[1]) * rgb.shape[0])
        frame = cv2.resize(
            rgb,
            (frame_width, new_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = add_step_stats_on_img(
            frame,
            offset,
            offset_mode,
            distance,
            distance_mode,
        )

    map_info = info.get("top_down_map_vlnce")
    if map_info is not None:
        top_down_map = map_info["map"]
        meters_per_px = map_info["meters_per_px"]
        bounds = map_info["bounds"]
        map_agent_pos = map_info["agent_map_coord"]
        mask = map_info["fog_of_war_mask"]
        rotation = map_info["agent_angle"]

        if not agent_stop and agent_action_elements is not None:
            maps.draw_waypoint_prediction(top_down_map, waypoint, meters_per_px, bounds)
        if oracle_waypoint is not None:
            maps.draw_oracle_waypoint(
                top_down_map, oracle_waypoint, meters_per_px, bounds
            )

        top_down_map = maps.colorize_topdown_map(
            top_down_map, mask, fog_of_war_desat_amount=0.75
        )

        top_down_map = habitat_maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=rotation,
            agent_radius_px=int(0.45 / meters_per_px),
        )

        if top_down_map.shape[1] < top_down_map.shape[0]:
            top_down_map = np.rot90(top_down_map, 1)

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map
        old_h, old_w, _ = top_down_map.shape
        top_down_width = 512 if frame is None else frame_width / 2
        top_down_height = int(top_down_width / old_w * old_h)
        top_down_map = cv2.resize(
            top_down_map,
            (int(top_down_width), top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )

        if frame is None:
            frame = top_down_map
        else:
            white = (
                np.ones(
                    (
                        top_down_map.shape[0],
                        frame.shape[1] - top_down_map.shape[1],
                        3,
                    )
                )
                * 255
            )
            add_instruction_on_img(white, observation["instruction_text"])
            map_and_inst = np.concatenate((white, top_down_map), axis=1)
            frame = np.concatenate((frame, map_and_inst), axis=0)

    return frame.astype(np.uint8)


def navigator_video_frame(
    observations,
    info,
    start_pos,
    start_heading,
    action=None,
    map_k="top_down_map_vlnce",
    frame_width=2048,
):
    def _rtheta_to_global_coordinates(r, theta, current_position, current_heading):
        phi = (current_heading + theta) % (2 * np.pi)
        x = current_position[0] - r * np.sin(phi)
        z = current_position[-1] - r * np.cos(phi)
        return [x, z]

    rgb = {k: v for k, v in observations.items() if k.startswith("rgb")}
    rgb["rgb_0"] = rgb["rgb"]
    del rgb["rgb"]
    rgb = [f[1] for f in sorted(rgb.items(), key=lambda f: int(f[0].split("_")[1]))]

    rgb = [
        add_id_on_img(rgb[i][:, 80 : (rgb[i].shape[1] - 80), :], str(i))
        for i in range(len(rgb))
    ][::-1]
    rgb = np.concatenate(rgb[6:] + rgb[:6], axis=1).astype(np.uint8)
    new_height = int((frame_width / rgb.shape[1]) * rgb.shape[0])
    rgb = cv2.resize(
        rgb,
        (frame_width, new_height),
        interpolation=cv2.INTER_CUBIC,
    )

    top_down_map = deepcopy(info[map_k]["map"])

    if action is not None and "action_args" in action:
        maps.draw_waypoint_prediction(
            top_down_map,
            _rtheta_to_global_coordinates(
                action["action_args"]["r"],
                action["action_args"]["theta"],
                start_pos,
                heading_from_quaternion(start_heading),
            ),
            info[map_k]["meters_per_px"],
            info[map_k]["bounds"],
        )

    top_down_map = maps.colorize_topdown_map(
        top_down_map,
        info[map_k]["fog_of_war_mask"],
        fog_of_war_desat_amount=0.75,
    )
    map_agent_pos = info[map_k]["agent_map_coord"]
    top_down_map = habitat_maps.draw_agent(
        image=top_down_map,
        agent_center_coord=map_agent_pos,
        agent_rotation=info[map_k]["agent_angle"],
        agent_radius_px=int(0.45 / info[map_k]["meters_per_px"]),
    )
    if top_down_map.shape[1] < top_down_map.shape[0]:
        top_down_map = np.rot90(top_down_map, 1)

    if top_down_map.shape[0] > top_down_map.shape[1]:
        top_down_map = np.rot90(top_down_map, 1)

    # scale top down map
    old_h, old_w, _ = top_down_map.shape
    top_down_height = rgb.shape[0]
    top_down_width = int(old_w * (top_down_height / old_h))
    top_down_map = cv2.resize(
        top_down_map,
        (int(top_down_width), top_down_height),
        interpolation=cv2.INTER_CUBIC,
    )

    inst_white = (
        np.ones((top_down_map.shape[0], rgb.shape[1] - top_down_map.shape[1], 3)) * 255
    )
    add_instruction_on_img(inst_white, observations["instruction"]["text"])
    map_and_inst = np.concatenate((inst_white, top_down_map), axis=1)
    horizontal_white = np.ones((50, rgb.shape[1], 3)) * 255
    return np.concatenate((rgb, horizontal_white, map_and_inst), axis=0).astype(
        np.uint8
    )


def compute_heading_to(
    pos_from: Union[list[float], ndarray], pos_to: Union[list[float], ndarray]
) -> tuple[list[float], float]:
    """Compute the heading that points from position `pos_from` to position `pos_to`
    in the global XZ coordinate frame.

    Args:
        pos_from: [x,y,z] or [x,z]
        pos_to: [x,y,z] or [x,z]

    Returns:
        heading quaternion as [x, y, z, w]
        heading scalar angle
    """
    delta_x = pos_to[0] - pos_from[0]
    delta_z = pos_to[-1] - pos_from[-1]
    xz_angle = np.arctan2(delta_x, delta_z)
    xz_angle = (xz_angle + np.pi) % (2 * np.pi)
    quat = quaternion_to_list(quaternion.from_euler_angles([0.0, xz_angle, 0.0]))
    return quat, xz_angle


def heading_from_quaternion(quat: quaternion.quaternion) -> float:
    # https://github.com/facebookresearch/habitat-lab/blob/v0.1.7/habitat/tasks/nav/nav.py#L356
    heading_vector = quaternion_rotate_vector(quat.inverse(), np.array([0, 0, -1]))
    phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
    return phi % (2 * np.pi)


def predictions_to_global_coordinates(
    pano: Tensor,
    offset: Tensor,
    distance: Tensor,
    current_position: Tensor,
    current_heading: Tensor,
    num_panos: int = 12,
) -> Tensor:
    """Takes a batch of waypoint predictions and converts them to global 2D
    Cartesian coordinates. `current_position` and `current_heading` are in the
    global XZ plane.
    Args:
        pano: Size([B])
        offset: Size([B])
        distance: Size([B])
        current_position: Size([B, 2]) or Size([B, 3])
        current_heading: Size([B])
    Returns:
        tensor of (x, z) coordinates of shape [Batch, 2]
    """
    radians_per_pano = (2 * np.pi) / num_panos
    relative_pano_center = pano * radians_per_pano
    phi = (current_heading + relative_pano_center + offset) % (2 * np.pi)

    x = current_position[:, 0] - distance * torch.sin(phi)
    z = current_position[:, current_position.shape[1] - 1] - distance * torch.cos(phi)
    return torch.stack([x, z], dim=1)


def rtheta_to_global_coordinates(
    sim: Simulator,
    r: float,
    theta: float,
    y_delta: float = 0.0,
    dimensionality: int = 2,
) -> list[float]:
    """Maps relative polar coordinates from an agent position to an updated
    agent position. The returned position is not validated for navigability.
    """
    assert dimensionality in [2, 3]
    scene_node = sim.get_agent(0).scene_node
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    agent_state = sim.get_agent_state()
    rotation = habitat_sim.utils.quat_from_angle_axis(theta, habitat_sim.geo.UP)
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)
    position = agent_state.position + (move_ax * r)
    position[1] += y_delta

    if dimensionality == 2:
        return [position[0], position[2]]
    return position
