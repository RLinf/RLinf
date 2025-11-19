import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import yaml
from dacite import from_dict

import numpy as np
import torch

import sapien

from mani_skill.utils.structs import Pose


"""
Task Planner Dataclasses
"""

PointTuple = Union[Tuple[float, float, float], List[float]]
RectCorners = Union[
    Tuple[PointTuple, PointTuple, PointTuple, PointTuple], List[PointTuple]
]
HandleJointIdxAndRelativeHandlePosition = Tuple[int, PointTuple]


@dataclass
class ArticulationConfig:
    articulation_type: str
    articulation_id: str
    articulation_handle_link_idx: int
    articulation_handle_active_joint_idx: int


@dataclass
class Subtask:
    type: str = field(init=False)
    uid: str = field(init=False)
    composite_subtask_uids: List[str] = field(init=False)

    def __post_init__(self):
        assert self.type in ["pick", "place", "navigate", "open", "close"]
        if getattr(self, "uid", None) is None:
            self.uid = self.type + "_" + "000000"
        if getattr(self, "composite_subtask_uids", None) is None:
            self.composite_subtask_uids = [self.uid]


@dataclass
class SubtaskConfig:
    task_id: int
    horizon: int = 200
    robot_cumulative_force_limit: float = torch.inf
    ee_rest_thresh: float = 0.05
    robot_resting_qpos_tolerance: float = 0.2
    robot_resting_qpos_tolerance_grasping: float = 0.6

    def __post_init__(self):
        assert self.horizon > 0
        assert self.ee_rest_thresh >= 0
        assert self.robot_resting_qpos_tolerance >= 0
        assert self.robot_resting_qpos_tolerance_grasping >= 0

    def update(self, update_dict: Dict):
        for k, v in update_dict.items():
            if getattr(self, k, None) is not None:
                setattr(self, k, v)
        return self


@dataclass
class PickSubtask(Subtask):
    obj_id: str
    articulation_config: Optional[ArticulationConfig] = None

    def __post_init__(self):
        self.type = "pick"
        super().__post_init__()


@dataclass
class PickSubtaskConfig(SubtaskConfig):
    task_id: int = 0
    robot_cumulative_force_limit: float = 5000


@dataclass
class PlaceSubtask(Subtask):
    obj_id: str
    goal_rectangle_corners: Optional[
        Union[List[str], RectCorners, List[RectCorners]]
    ] = None
    goal_pos: Optional[Union[PointTuple, List[PointTuple], str]] = None
    validate_goal_rectangle_corners: bool = True
    articulation_config: Optional[ArticulationConfig] = None

    def __post_init__(self):
        self.type = "place"
        super().__post_init__()
        if (
            self.validate_goal_rectangle_corners
            and self.goal_rectangle_corners is not None
        ):
            self.goal_rectangle_corners = self._parse_rect_corners(
                self.goal_rectangle_corners
            )

        if isinstance(self.goal_pos, str):
            self.goal_pos = [float(coord) for coord in self.goal_pos.split(",")]

    def _parse_rect_corners(self, rect_corners):
        for i, corner in enumerate(rect_corners):
            if isinstance(corner, str):
                rect_corners[i] = [float(coord) for coord in corner.split(",")]
        # make sure have exactly 4 corners at the same height
        assert len(rect_corners) == 4, "Goal rectangle must have exactly 4 corners"
        A, B, C, D = [np.array(corner) for corner in rect_corners]
        sides0 = np.array([B - A, C - B, D - C, A - D])
        sides1 = np.array([D - A, A - B, B - C, C - D])
        points_angles = np.rad2deg(
            np.arccos(
                np.sum(sides0 * sides1, axis=1)
                / (np.linalg.norm(sides0, axis=1) * np.linalg.norm(sides1, axis=1))
            )
        )
        assert np.all(
            np.abs(points_angles - 90) < 1e-3
        ), f"Should have points in ABCD order, but got angles {points_angles} between sides AB/AD, BC/BA, CD/CB, DA/DC"
        return rect_corners


@dataclass
class PlaceSubtaskConfig(SubtaskConfig):
    task_id: int = 1
    obj_goal_thresh: float = 0.15
    goal_type: str = "sphere"
    robot_cumulative_force_limit: float = 7500

    def __post_init__(self):
        super().__post_init__()
        assert self.obj_goal_thresh >= 0
        # cylinder means cylindrical goal centered at place_subtask.goal_pos
        # zone means use place_subtask.goal_rectangle_corners to establish rectangular zone
        assert self.goal_type in ["zone", "cylinder", "sphere"]


@dataclass
class NavigateSubtask(Subtask):
    obj_id: Optional[str] = None
    goal_pos: Optional[PointTuple] = None
    prev_goal_pos: Optional[PointTuple] = None
    articulation_config: Optional[ArticulationConfig] = None
    # NOTE (arth): see note in OpenSubtask
    remove_obj_id: Optional[str] = None

    def __post_init__(self):
        self.type = "navigate"
        super().__post_init__()


@dataclass
class NavigateSubtaskConfig(SubtaskConfig):
    task_id: int = 2
    horizon: int = 500
    navigated_successfully_dist: float = 2
    navigated_successfully_rot: float = 0.5
    ignore_arm_checkers: bool = False


@dataclass
class OpenSubtask(Subtask, ArticulationConfig):
    obj_id: str
    articulation_relative_handle_pos: Union[PointTuple, sapien.Pose, Pose]

    def __post_init__(self):
        self.type = "open"
        super().__post_init__()


@dataclass
class OpenSubtaskConfig(SubtaskConfig):
    task_id: int = 3
    robot_cumulative_force_limit: float = 10_000
    joint_qpos_open_thresh_frac: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        super().__post_init__()

        assert isinstance(self.joint_qpos_open_thresh_frac, dict)
        assert (
            "default" in self.joint_qpos_open_thresh_frac
        ), "joint_qpos_open_thresh_frac requires default value to cover cases where a different articulation is opened"
        for v in self.joint_qpos_open_thresh_frac.values():
            assert (
                isinstance(v, float) and 0 <= v and v <= 1
            ), f"joint_qpos_open_thresh_frac should be a float in [0, 1], instead got {v}"


@dataclass
class CloseSubtask(Subtask, ArticulationConfig):
    articulation_relative_handle_pos: Union[PointTuple, sapien.Pose, Pose]
    # NOTE (arth): this is somewhat of a band-aid solution to easily
    #       remove the bowl inside the kitchen_counter drawer
    #       in a future version, maybe can make a list of objects to
    #       teleport to specific location in gen_spawn_positions.py
    remove_obj_id: Optional[str] = None

    def __post_init__(self):
        self.type = "close"
        super().__post_init__()


@dataclass
class CloseSubtaskConfig(SubtaskConfig):
    task_id: int = 4
    robot_cumulative_force_limit: float = 10_000
    joint_qpos_close_thresh_frac: float = 0.01


@dataclass
class TaskPlan:
    subtasks: List[Subtask]
    build_config_name: Optional[str] = None
    init_config_name: Optional[str] = None


"""
Reading Task Plan from file
"""


@dataclass
class PlanData:
    dataset: str
    plans: List[TaskPlan]


def plan_data_from_file(config_path: str = None) -> PlanData:
    config_path: Path = Path(config_path)
    assert config_path.exists(), f"Path {config_path} not found"

    suffix = Path(config_path).suffix
    if suffix == ".json":
        with open(config_path, "rb") as f:
            plan_data = json.load(f)
    elif suffix == ".yml":
        with open(config_path) as f:
            plan_data = yaml.safe_load(f)
    else:
        print(f"{suffix} not supported")

    plans = []
    for task_plan_data in plan_data["plans"]:
        build_config_name = task_plan_data["build_config_name"]
        init_config_name = task_plan_data["init_config_name"]
        subtasks = []
        for subtask in task_plan_data["subtasks"]:
            subtask_type = subtask["type"]
            if subtask_type == "pick":
                cls = PickSubtask
            elif subtask_type == "place":
                cls = PlaceSubtask
            elif subtask_type == "navigate":
                cls = NavigateSubtask
            elif subtask_type == "open":
                cls = OpenSubtask
            elif subtask_type == "close":
                cls = CloseSubtask
            else:
                raise NotImplementedError(f"Subtask {subtask_type} not implemented yet")
            subtasks.append(from_dict(data_class=cls, data=subtask))
        plans.append(
            TaskPlan(
                subtasks=subtasks,
                build_config_name=build_config_name,
                init_config_name=init_config_name,
            )
        )

    return PlanData(dataset=plan_data["dataset"], plans=plans)
