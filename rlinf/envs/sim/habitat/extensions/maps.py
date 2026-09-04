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

from typing import Optional, Union

import networkx as nx
import numpy as np
from habitat.core.simulator import Simulator

try:
    from habitat.core.utils import try_cv2_import

    cv2 = try_cv2_import()
except ImportError:
    import cv2
from habitat.tasks.vln.vln import VLNEpisode
from habitat.utils.visualizations import maps as habitat_maps

AGENT_SPRITE = habitat_maps.AGENT_SPRITE

MAP_THICKNESS_SCALAR: int = 128

MAP_INVALID_POINT = 0
MAP_VALID_POINT = 1
MAP_BORDER_INDICATOR = 2
MAP_SOURCE_POINT_INDICATOR = 4
MAP_TARGET_POINT_INDICATOR = 6
MAP_MP3D_WAYPOINT = 7
MAP_VIEW_POINT_INDICATOR = 8
MAP_TARGET_BOUNDING_BOX = 9
MAP_REFERENCE_POINT = 10
MAP_MP3D_REFERENCE_PATH = 11
MAP_WAYPOINT_PREDICTION = 12
MAP_ORACLE_WAYPOINT = 13
MAP_SHORTEST_PATH_WAYPOINT = 14

TOP_DOWN_MAP_COLORS = np.full((256, 3), 150, dtype=np.uint8)
TOP_DOWN_MAP_COLORS[15:] = cv2.applyColorMap(
    np.arange(241, dtype=np.uint8), cv2.COLORMAP_JET
).squeeze(1)[:, ::-1]
TOP_DOWN_MAP_COLORS[MAP_INVALID_POINT] = [255, 255, 255]  # White
TOP_DOWN_MAP_COLORS[MAP_VALID_POINT] = [150, 150, 150]  # Light Grey
TOP_DOWN_MAP_COLORS[MAP_BORDER_INDICATOR] = [50, 50, 50]  # Grey
TOP_DOWN_MAP_COLORS[MAP_SOURCE_POINT_INDICATOR] = [0, 0, 200]  # Blue
TOP_DOWN_MAP_COLORS[MAP_TARGET_POINT_INDICATOR] = [200, 0, 0]  # Red
TOP_DOWN_MAP_COLORS[MAP_MP3D_WAYPOINT] = [0, 200, 0]  # Green
TOP_DOWN_MAP_COLORS[MAP_VIEW_POINT_INDICATOR] = [245, 150, 150]  # Light Red
TOP_DOWN_MAP_COLORS[MAP_TARGET_BOUNDING_BOX] = [0, 175, 0]  # Dark Green
TOP_DOWN_MAP_COLORS[MAP_REFERENCE_POINT] = [0, 0, 0]  # Black
TOP_DOWN_MAP_COLORS[MAP_MP3D_REFERENCE_PATH] = [0, 0, 0]  # Black
TOP_DOWN_MAP_COLORS[MAP_WAYPOINT_PREDICTION] = [255, 255, 0]  # Yellow
TOP_DOWN_MAP_COLORS[MAP_ORACLE_WAYPOINT] = [255, 165, 0]  # Orange
TOP_DOWN_MAP_COLORS[MAP_SHORTEST_PATH_WAYPOINT] = [0, 150, 0]  # Dark Green


def get_top_down_map(sim, map_resolution, meters_per_pixel):
    base_height = sim.get_agent(0).state.position[1]
    td_map = habitat_maps.get_topdown_map(
        sim.pathfinder,
        base_height,
        map_resolution,
        False,
        meters_per_pixel,
    )
    return td_map


def colorize_topdown_map(
    top_down_map: np.ndarray,
    fog_of_war_mask: Optional[np.ndarray] = None,
    fog_of_war_desat_amount: float = 0.5,
) -> np.ndarray:
    """Same as `maps.colorize_topdown_map` in Habitat-Lab, but with different
    colors.
    """
    _map = TOP_DOWN_MAP_COLORS[top_down_map]

    if fog_of_war_mask is not None:
        fog_of_war_desat_values = np.array([[fog_of_war_desat_amount], [1.0]])
        # Only desaturate valid points as only valid points get revealed
        desat_mask = top_down_map != MAP_INVALID_POINT

        _map[desat_mask] = (_map * fog_of_war_desat_values[fog_of_war_mask]).astype(
            np.uint8
        )[desat_mask]

    return _map


def static_to_grid(
    realworld_x: float,
    realworld_y: float,
    grid_resolution: tuple[int, int],
    bounds: dict[str, tuple[float, float]],
) -> tuple[int, int]:
    """Return gridworld index of realworld coordinates assuming top-left
    corner is the origin. The real world coordinates of lower left corner are
    (coordinate_min, coordinate_min) and of top right corner are
    (coordinate_max, coordinate_max). Same as the habitat-Lab maps.to_grid
    function but with a static `bounds` instead of requiring a simulator or
    pathfinder instance.
    """
    grid_size = (
        abs(bounds["upper"][2] - bounds["lower"][2]) / grid_resolution[0],
        abs(bounds["upper"][0] - bounds["lower"][0]) / grid_resolution[1],
    )
    grid_x = int((realworld_x - bounds["lower"][2]) / grid_size[0])
    grid_y = int((realworld_y - bounds["lower"][0]) / grid_size[1])
    return grid_x, grid_y


def drawline(
    img: np.ndarray,
    pt1: Union[tuple[float], list[float]],
    pt2: Union[tuple[float], list[float]],
    color: list[int],
    thickness: int = 1,
    style: str = "dotted",
    gap: int = 15,
) -> None:
    """https://stackoverflow.com/questions/26690932/opencv-rectangle-with-dotted-or-dashed-lines
    style: "dotted", "dashed", or "filled"
    """
    assert style in ["dotted", "dashed", "filled"]

    if style == "filled":
        cv2.line(img, pt1, pt2, color, thickness)
        return

    dist = ((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2) ** 0.5
    pts = []
    for i in np.arange(0, dist, gap):
        r = i / dist
        x = int((pt1[0] * (1 - r) + pt2[0] * r) + 0.5)
        y = int((pt1[1] * (1 - r) + pt2[1] * r) + 0.5)
        pts.append((x, y))

    if style == "dotted":
        for p in pts:
            cv2.circle(img, p, thickness, color, -1)
    else:
        s = pts[0]
        e = pts[0]
        for i, p in enumerate(pts):
            s = e
            e = p
            if i % 2 == 1:
                cv2.line(img, s, e, color, thickness)


def drawpoint(
    img: np.ndarray,
    position: Union[tuple[int], list[int]],
    color: list[int],
    meters_per_px: float,
    pad: float = 0.3,
) -> None:
    point_padding = int(pad / meters_per_px)
    img[
        position[0] - point_padding : position[0] + point_padding + 1,
        position[1] - point_padding : position[1] + point_padding + 1,
    ] = color


def draw_triangle(
    img: np.ndarray,
    centroid: Union[tuple[int], list[int]],
    color: list[int],
    meters_per_px: float,
    pad: int = 0.35,
) -> None:
    point_padding = int(pad / meters_per_px)

    # (Y, X)
    left = (centroid[1] - point_padding, centroid[0] + point_padding)
    right = (centroid[1] + point_padding, centroid[0] + point_padding)
    top = (centroid[1], centroid[0] - point_padding)
    cv2.drawContours(img, [np.array([left, right, top])], 0, color, -1)


def draw_reference_path(
    img: np.ndarray,
    sim: Simulator,
    episode: VLNEpisode,
    map_resolution: int,
    meters_per_px: float,
) -> None:
    """Draws lines between each waypoint in the reference path."""
    shortest_path_points = [
        habitat_maps.to_grid(
            p[2],
            p[0],
            img.shape[0:2],
            sim,
        )[::-1]
        for p in episode.reference_path
    ]

    pt_from = None
    for i, pt_to in enumerate(shortest_path_points):
        if i != 0:
            drawline(
                img,
                (pt_from[0], pt_from[1]),
                (pt_to[0], pt_to[1]),
                MAP_SHORTEST_PATH_WAYPOINT,
                thickness=int(0.4 * map_resolution / MAP_THICKNESS_SCALAR),
                style="dashed",
                gap=10,
            )
        pt_from = pt_to

    for pt in shortest_path_points:
        drawpoint(img, (pt[1], pt[0]), MAP_SHORTEST_PATH_WAYPOINT, meters_per_px)


def draw_straight_shortest_path_points(
    img: np.ndarray,
    sim: Simulator,
    map_resolution: int,
    shortest_path_points: list[list[float]],
) -> None:
    """Draws the shortest path from start to goal assuming a standard
    discrete action space.
    """
    shortest_path_points = [
        habitat_maps.to_grid(p[2], p[0], img.shape[0:2], sim)[::-1]
        for p in shortest_path_points
    ]

    habitat_maps.draw_path(
        img,
        [(p[1], p[0]) for p in shortest_path_points],
        MAP_SHORTEST_PATH_WAYPOINT,
        int(0.4 * map_resolution / MAP_THICKNESS_SCALAR),
    )


def draw_source_and_target(
    img: np.ndarray, sim: Simulator, episode: VLNEpisode, meters_per_px: float
) -> None:
    s_x, s_y = habitat_maps.to_grid(
        episode.start_position[2],
        episode.start_position[0],
        img.shape[0:2],
        sim,
    )
    drawpoint(img, (s_x, s_y), MAP_SOURCE_POINT_INDICATOR, meters_per_px)

    # mark target point
    t_x, t_y = habitat_maps.to_grid(
        episode.goals[0].position[2],
        episode.goals[0].position[0],
        img.shape[0:2],
        sim,
    )
    drawpoint(img, (t_x, t_y), MAP_TARGET_POINT_INDICATOR, meters_per_px)


def draw_waypoint_prediction(
    img: np.ndarray,
    waypoint: Union[tuple[float], list[float]],
    meters_per_px: float,
    bounds: dict[str, tuple[float]],
) -> None:
    w_x, w_y = static_to_grid(waypoint[1], waypoint[0], img.shape[0:2], bounds)
    if w_x < img.shape[0] and w_x > 0 and w_y < img.shape[1] and w_y > 0:
        draw_triangle(img, (w_x, w_y), MAP_WAYPOINT_PREDICTION, meters_per_px)


def draw_oracle_waypoint(
    img: np.ndarray,
    waypoint: Union[tuple[float], list[float]],
    meters_per_px: float,
    bounds: dict[str, tuple[float]],
) -> None:
    w_x, w_y = static_to_grid(waypoint[1], waypoint[0], img.shape[0:2], bounds)
    draw_triangle(img, (w_x, w_y), MAP_ORACLE_WAYPOINT, meters_per_px, pad=0.2)


def get_nearest_node(graph: nx.Graph, current_position: list[float]) -> str:
    """Determine the closest MP3D node to the agent's start position as given
    by a [x,z] position vector.
    Returns:
        node ID
    """
    nearest = None
    dist = float("inf")
    for node in graph:
        node_pos = graph.nodes[node]["position"]
        node_pos = np.take(node_pos, (0, 2))
        cur_dist = np.linalg.norm(
            np.array(node_pos) - np.array(current_position), ord=2
        )
        if cur_dist < dist:
            dist = cur_dist
            nearest = node
    return nearest


def update_nearest_node(
    graph: nx.Graph, nearest_node: str, current_position: np.ndarray
) -> str:
    """Determine the closest MP3D node to the agent's current position as
    given by a [x,z] position vector. The selected node must be reachable
    from the previous MP3D node as specified in the nav-graph edges.
    Returns:
        node ID
    """
    nearest = None
    dist = float("inf")

    for node in [nearest_node] + [e[1] for e in graph.edges(nearest_node)]:
        node_pos = graph.nodes[node]["position"]
        node_pos = np.take(node_pos, (0, 2))
        cur_dist = np.linalg.norm(
            np.array(node_pos) - np.array(current_position), ord=2
        )
        if cur_dist < dist:
            dist = cur_dist
            nearest = node
    return nearest


def draw_mp3d_nodes(
    img: np.ndarray,
    sim: Simulator,
    episode: VLNEpisode,
    graph: nx.Graph,
    meters_per_px: float,
) -> None:
    n = get_nearest_node(graph, (episode.start_position[0], episode.start_position[2]))
    starting_height = graph.nodes[n]["position"][1]
    for node in graph:
        pos = graph.nodes[node]["position"]

        # no obvious way to differentiate between floors. Use this for now.
        if abs(pos[1] - starting_height) < 1.0:
            r_x, r_y = habitat_maps.to_grid(pos[2], pos[0], img.shape[0:2], sim)

            # only paint if over a valid point
            if img[r_x, r_y]:
                drawpoint(img, (r_x, r_y), MAP_MP3D_WAYPOINT, meters_per_px)
