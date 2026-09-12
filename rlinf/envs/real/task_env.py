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

"""The Gymnasium env that runs one task on one robot.

:class:`TaskEnv` takes a composed :class:`~rlinf.robotics.Robot`, a
:class:`~rlinf.envs.real.tasks.Task`, and the two halves of the policy's
contract -- an :class:`~rlinf.envs.real.policy.ActionLayout` and an
:class:`~rlinf.envs.real.policy.ObservationSpec` -- and owns the step loop,
episode bookkeeping, and the robot's lifecycle. Nothing robot-specific lives
here.

:class:`RegisteredTaskEnv` builds those from the ``override_cfg`` and hardware
a run is given, so a Gymnasium id can name one. Each robot package subclasses
it once to say how its robot is driven and what its policy reads, and each
task id is then one more line on top.
"""

import copy
import dataclasses
import queue
import time
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.real.policy import ActionLayout, Applied, ObservationSpec
from rlinf.envs.real.tasks import Needs, Parts, Reading, ResetContext, Task, bind
from rlinf.envs.real.tasks.requirements import at, combine
from rlinf.envs.real.utils.config import get_hardware_config
from rlinf.envs.real.utils.frames import crop_region, policy_depth, policy_frame
from rlinf.envs.real.utils.reward_model import RewardModel
from rlinf.envs.real.utils.seeding import seed_sampled_spaces
from rlinf.envs.real.utils.video import VideoPlayer
from rlinf.robotics import Arm, Camera, Robot
from rlinf.robotics.actions import ActionKind, ActionPart
from rlinf.robotics.discovery import RobotConfig, RobotDiscovery, RobotInfo
from rlinf.robotics.parts.cameras import CameraInfo
from rlinf.scheduler import WorkerInfo
from rlinf.utils.logging import get_logger


@dataclass
class TaskEnvConfig:
    """How an episode runs, whatever the task and robot."""

    is_dummy: bool = False
    """Run without hardware, sampling observations from the space."""

    step_frequency: float = 10.0
    """Control rate in Hz. A step sleeps for the remainder of its period."""

    max_num_steps: int = 100
    """Steps before an episode is truncated."""

    reward_scale: float = 1.0
    """Multiplier on the reward, applied after the termination check."""

    enable_camera_player: bool = True
    """Show the cameras in a viewer window."""


@dataclass
class RegisteredTaskEnvConfig(TaskEnvConfig):
    """What a registered id adds: camera layout and a learned reward."""

    camera_names: Optional[Mapping[str, str]] = None
    """Name each camera by serial. Unnamed cameras are ``wrist_1``,
    ``wrist_2``, and so on, in the hardware's order."""

    camera_crop_regions: Optional[Mapping[str, Sequence[float]]] = None
    """``[top, left, bottom, right]`` fractions each camera is cropped to, by
    serial. Uncropped cameras keep their centred square."""

    enable_camera_depth: bool = False
    """Capture depth beside every colour frame and hand it to the policy."""

    use_reward_model: bool = False
    """Score steps with a reward worker instead of the task."""

    reward_worker_cfg: Optional[dict] = None
    """The reward worker's config. The env worker fills this in."""

    reward_worker_hardware_rank: Optional[int] = None
    """Accelerator the reward worker runs on."""

    reward_worker_node_rank: Optional[int] = None
    """Node the reward worker runs on; ``None`` runs it beside the env."""

    reward_worker_node_group: Optional[str] = None
    """Node group the reward worker is placed in."""

    reward_image_key: Optional[str] = None
    """Camera the reward model scores; ``None`` takes the first by name."""

    def __post_init__(self) -> None:
        if self.camera_names is not None:
            self.camera_names = {
                str(serial): str(name) for serial, name in self.camera_names.items()
            }
        if self.camera_crop_regions is not None:
            self.camera_crop_regions = {
                str(serial): region
                for serial, region in self.camera_crop_regions.items()
            }


class TaskEnv(gym.Env):
    """Run ``task`` on ``robot``, driven by the policy's action layout.

    Construction checks that the robot has what the task, the action layout,
    and the observation need, before connecting it, then connects it, waits
    for its arms, and homes it. A dummy env takes no robot and samples its
    observation space instead.

    Args:
        robot: The composed robot, connected or not; ``None`` when dummy.
        task: What the episode is for.
        action: The channels a policy's action is cut into.
        observation: What the policy reads.
        config: How episodes run.
        reward_model: Scores each step's policy frames in place of the task
            when given, such as a :class:`RewardModel`.
    """

    metadata = {"render_modes": []}

    #: Teleoperation devices whose action layout fits this env.
    TELEOP: ClassVar[tuple[str, ...]] = ()

    #: Device used when a run names none.
    TELEOP_DEFAULT: ClassVar[str] = "none"

    #: Whole-robot reads tried before a stalled camera ends the step.
    READ_ATTEMPTS: ClassVar[int] = 3

    #: Seconds between those reads, for a camera to come back.
    STALL_WAIT_S: ClassVar[float] = 5.0

    def __init__(
        self,
        robot: Optional[Robot],
        task: Task,
        action: ActionLayout,
        *,
        observation: ObservationSpec,
        config: Optional[TaskEnvConfig] = None,
        reward_model: Optional[Callable[[Mapping[str, np.ndarray]], float]] = None,
    ) -> None:
        self.config = config if config is not None else TaskEnvConfig()
        if robot is None and not self.config.is_dummy:
            raise ValueError("A TaskEnv needs a robot unless it is dummy.")
        self._logger = get_logger()
        self.robot = robot
        self.task = task
        self.action = action
        self.observation = observation
        self.reward_model = reward_model
        task.validate(action.dof())
        action.confine(task.workspace)

        self.action_space = action.space()
        self.observation_space = self._observation_space()
        # A wrapper may rewrite the space it is handed in place; a dummy env
        # samples this private copy so its observations keep the raw layout.
        self._sample_space = copy.deepcopy(self.observation_space)
        self._num_steps = 0
        self._hold = 0
        self._in_zone = False
        self._reading: Optional[Mapping[str, Any]] = None
        self.parts: Optional[Parts] = None
        self.camera_player: Optional[VideoPlayer] = None
        self._camera_paths: Mapping[str, str] = {}
        if self.config.is_dummy:
            return

        self.parts = bind(robot, self._requirements(), owner=type(task).__name__)
        if not robot.is_connected:
            robot.connect()
        self._wait_for_arms()
        # A camera composed inside a group, as a wrist camera is, reports under
        # that group rather than at the top of the reading.
        self._camera_paths = {
            path.rsplit(".", 1)[-1]: path for path in robot.parts_of_type(Camera)
        }
        self.camera_player = VideoPlayer(self.config.enable_camera_player)
        task.home(self.parts, self._context())
        self._reading = self._read()

    # Protocol the wrapper stack and teleop read off the unwrapped env.

    @property
    def ACTION_WRAPPERS(self) -> tuple[str, ...]:  # noqa: N802 - wrapper protocol
        """Action wrappers that fit this action layout."""
        return self.action.wrappers

    @property
    def TRANSFORMS(self) -> tuple[str, ...]:  # noqa: N802 - wrapper protocol
        """Observation and action transforms that fit it."""
        return self.action.transforms

    @property
    def task_description(self) -> str:
        """Language instruction for policies that condition on one."""
        return self.task.description

    @property
    def num_steps(self) -> int:
        """Steps taken in the current episode."""
        return self._num_steps

    def action_parts(self) -> tuple[ActionPart, ...]:
        """Named slices of the action, for teleop and action wrappers."""
        return self.action.parts()

    # Context teleoperation devices read to line their commands up with the
    # action. A getter returns ``None`` where the control has no such thing.

    def get_joint_positions(self) -> Optional[np.ndarray]:
        """Joints of each driven arm as ``(arms, dof)``, zeros before a read.

        ``None`` when no channel commands joints.
        """
        dofs = self.action.dof()
        if not dofs:
            return None
        rows = []
        for role, dof in dofs.items():
            joints = None
            if self._reading is not None and self.parts is not None:
                joints = (
                    self.parts.read(self._reading).arm(role).get("arm_joint_position")
                )
            rows.append(np.zeros(dof or 0) if joints is None else np.asarray(joints))
        return np.stack(rows).astype(float)

    def get_tcp_pose(self) -> Optional[np.ndarray]:
        """The tool pose the next Cartesian delta is applied to, read now.

        Reading here also moves the base of the next step's delta to this
        pose, so a device's delta and the env's agree on where they start.
        ``None`` when the action is not a Cartesian delta.
        """
        roles = [
            channel.role
            for channel in self.action.channels
            if channel.kind is ActionKind.CARTESIAN_DELTA
        ]
        if not roles or self.parts is None:
            return None
        self._reading = self._read()
        return np.asarray(self.parts.read(self._reading).arm(roles[0])["tcp_pose"])

    def get_action_scale(self) -> Optional[np.ndarray]:
        """What one unit of each action channel moves."""
        return self.action.teleop_context(self.parts).get("action_scale")

    def refresh_reading(self) -> Optional[Reading]:
        """Read the robot again and keep it, for a device that moved it itself.

        A streaming teleoperation device commands the arms between steps, so
        the reading this env cached is behind the robot. Reading here brings
        the two back together before the next step scores anything.

        Returns:
            The new reading by role, or ``None`` with no robot to read.
        """
        if self.parts is None or self.config.is_dummy:
            return None
        self._reading = self._read()
        return self.parts.read(self._reading)

    def get_gripper_open(self) -> Optional[bool]:
        """Whether the gripper is open, for a device that toggles it."""
        return self.action.teleop_context(self.parts).get("gripper_open")

    def get_hand_reset_pose(self) -> Optional[np.ndarray]:
        """The hand's resting finger pose."""
        return self.action.teleop_context(self.parts).get("hand_reset_pose")

    # Gymnasium API.

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Put the robot and scene back and return the first observation."""
        super().reset(seed=seed)
        # A dummy env samples its space instead of reading, so seeding the
        # space is what makes such a run reproducible.
        seed_sampled_spaces(seed, self._sample_space)
        self._num_steps = 0
        self._hold = 0
        self._in_zone = False
        self.action.reset(self.parts)
        if self.config.is_dummy:
            return self._sample_space.sample(), {}
        self.task.reset(self.parts, self._context(options))
        return self._observe(), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Apply one action, wait out the control period, and score the result."""
        start = time.time()
        action = np.clip(action, self.action_space.low, self.action_space.high)
        applied = Applied()
        if not self.config.is_dummy:
            applied = self.action.apply(
                self.parts, action, self.parts.read(self._reading or {})
            )
        self._num_steps += 1
        time.sleep(max(0.0, 1.0 / self.config.step_frequency - (time.time() - start)))

        if self.config.is_dummy:
            observation, reward = self._sample_space.sample(), 0.0
        else:
            observation = self._observe()
            reward = self._score(applied, observation)
        terminated = reward >= 1.0 and self._hold >= self.task.config.success_hold_steps
        truncated = self._num_steps >= self.config.max_num_steps
        # Whether this step reached the goal is the env's to report, so a
        # runner does not have to guess it back out of the reward it scaled.
        info = {"in_zone": self._in_zone}
        return (
            observation,
            reward * self.config.reward_scale,
            terminated,
            truncated,
            info,
        )

    def close(self) -> None:
        """Stop the viewer and disconnect the robot."""
        if self.camera_player is not None:
            self.camera_player.stop()
            self.camera_player = None
        # Let a deferred command reach the robot before the robot goes away.
        self.action.close()
        if self.robot is not None:
            self.robot.disconnect()
            self.robot = None

    # Internals.

    def _observation_space(self) -> gym.spaces.Dict:
        spec = self.observation
        state = gym.spaces.Dict(
            {
                key.key: gym.spaces.Box(
                    key.low, key.high, shape=key.shape, dtype=key.dtype
                )
                for key in spec.state
            }
        )
        spaces: dict[str, gym.Space] = {"state": state}
        # Gymnasium's env checker rejects an empty Dict space, so a robot with
        # no camera reports no 'frames' key at all rather than an empty one.
        if spec.cameras:
            spaces["frames"] = gym.spaces.Dict(
                {
                    camera.name: gym.spaces.Box(
                        0, 255, shape=(*spec.frame_size, 3), dtype=np.uint8
                    )
                    for camera in spec.cameras
                }
            )
        depth = [camera for camera in spec.cameras if camera.enable_depth]
        if depth:
            spaces["depths"] = gym.spaces.Dict(
                {
                    camera.name: gym.spaces.Box(
                        0.0, np.inf, shape=spec.frame_size, dtype=np.float32
                    )
                    for camera in depth
                }
            )
        return gym.spaces.Dict(spaces)

    def _requirements(self) -> dict[str, Needs]:
        """What the task, the action layout, and the observation need together."""
        needs = combine(self.task.requirements(), self.action.requirements())
        for declared in self.observation.requirements():
            for role, need in declared.items():
                # A role the task or the layout already named keeps its part
                # category; the observation only adds fields to read.
                known = needs.get(role, Needs())
                needs[role] = known | Needs(
                    kind=known.kind,
                    observes=need.observes,
                    commands=need.commands,
                    end_effector=need.end_effector,
                )
        return needs

    def _context(self, options: Optional[Mapping[str, Any]] = None) -> ResetContext:
        return ResetContext(
            rng=self.np_random,
            action=self.action,
            options=options or {},
            rate_hz=self.config.step_frequency,
        )

    def _wait_for_arms(self) -> None:
        arms = [
            part
            for part in map(self.parts.part, self.parts.roles)
            if isinstance(part, Arm)
        ]
        start = warned = time.time()
        while not all(arm.is_robot_up() for arm in arms):
            time.sleep(0.5)
            if time.time() - warned > 30:
                warned = time.time()
                self._logger.warning(
                    "Waited %.0fs for the arm to report ready.", warned - start
                )

    def _read(self) -> Mapping[str, Any]:
        """Read the whole robot once, reopening a camera that has stalled."""
        for attempt in range(1, self.READ_ATTEMPTS):
            try:
                return self.robot.get_observation()
            except queue.Empty:
                self._logger.warning(
                    "A camera stopped producing frames; reopening it and reading "
                    "again in %.0fs (attempt %d of %d).",
                    self.STALL_WAIT_S,
                    attempt + 1,
                    self.READ_ATTEMPTS,
                )
                time.sleep(self.STALL_WAIT_S)
                # Reopen the camera in place rather than rebuilding it, which
                # would drop the placement the robot gave it.
                for camera in self.robot.parts_of_type(Camera).values():
                    if not camera.is_ready():
                        camera.reopen()
        return self.robot.get_observation()

    def _observe(self) -> dict[str, Any]:
        """Read the robot once and build the policy's observation from it."""
        self._reading = self._read()
        state = self.observation.read(self.parts.read(self._reading))
        observation: dict[str, Any] = {"state": state}
        if self.observation.cameras:
            frames, depths, display = {}, {}, {}
            size = self.observation.frame_size
            for camera in self.observation.cameras:
                captured = at(self._reading, self._camera_paths[camera.name])
                frame, cropped = policy_frame(
                    captured["frame"],
                    size,
                    camera.crop_region,
                    to_rgb=self.observation.to_rgb,
                )
                frames[camera.name] = frame
                # The viewer shows the camera's own colours either way.
                display[camera.name] = (
                    frame[..., ::-1] if self.observation.to_rgb else frame
                )
                display[f"{camera.name}_full"] = cropped
                if camera.enable_depth and "depth" in captured:
                    depths[camera.name] = policy_depth(
                        captured["depth"], size, camera.crop_region
                    )
            if self.camera_player is not None:
                self.camera_player.put_frame(display)
            observation["frames"] = frames
            if depths:
                observation["depths"] = depths
        return observation

    def _score(self, applied: Applied, observation: Mapping[str, Any]) -> float:
        """The step's reward, less any gripper penalty, and the success streak.

        A reward model, when there is one, scores the policy's frames in
        place of the task, and a step it scores 1 or more counts toward the
        streak.
        """
        if self.reward_model is not None:
            reward = self.reward_model(observation.get("frames", {}))
            in_zone = reward >= 1.0
        else:
            evaluation = self.task.evaluate(self.parts.read(self._reading), applied)
            reward, in_zone = evaluation.reward, evaluation.in_zone
        self._in_zone = bool(in_zone)
        self._hold = self._hold + 1 if in_zone else 0
        config = self.task.config
        if config.enable_gripper_penalty:
            reward -= config.gripper_penalty * applied.penalties
        return reward


def split_overrides(
    settings: Mapping[str, Any], owners: Sequence[type], *, owner: str
) -> tuple[Any, ...]:
    """Build one config per owner from a flat ``override_cfg``.

    Every key belongs to exactly one owner's dataclass. A key no owner takes is
    refused, as a dataclass refuses one, so a misspelled or misplaced setting
    fails here instead of being ignored.

    Args:
        settings: The run's ``override_cfg``, with defaults merged in.
        owners: Config dataclasses, each built from the keys it declares.
        owner: Who is being configured, for error messages.

    Raises:
        TypeError: If two owners declare one key, or no owner declares one.
    """
    claimed: dict[str, type] = {}
    for config_cls in owners:
        for field in dataclasses.fields(config_cls):
            if not field.init:
                continue
            if field.name in claimed:
                raise TypeError(
                    f"{owner}: {field.name!r} is declared by both "
                    f"{claimed[field.name].__name__} and {config_cls.__name__}."
                )
            claimed[field.name] = config_cls
    unknown = sorted(set(settings) - set(claimed))
    if unknown:
        raise TypeError(
            f"{owner}.__init__() got unexpected keyword arguments {unknown}."
        )
    return tuple(
        config_cls(
            **{
                key: value
                for key, value in settings.items()
                if claimed[key] is config_cls
            }
        )
        for config_cls in owners
    )


class RegisteredTaskEnv(TaskEnv):
    """A :class:`TaskEnv` built from a run's config, as a Gymnasium id needs.

    A robot package subclasses this once, saying how its robot is driven and
    what its policy reads. A task id is then a subclass that names the task
    and any defaults it differs by::

        class PiperReachEnv(PiperEnv):
            TASK = JointReach

    These classes only configure; ``step``, ``reset`` and the observation
    belong to :class:`TaskEnv` for every robot alike.
    """

    #: The robot class, composed from its registered hardware config.
    ROBOT: ClassVar[type[Robot]]

    #: Config dataclass for the settings this robot's action channels take.
    ACTION_CONFIG: ClassVar[type]

    #: The task this id runs.
    TASK: ClassVar[type[Task]]

    #: Settings a run gets unless it overrides them. Merged down the class
    #: hierarchy, so a task id adds to its robot preset's rather than
    #: replacing them.
    DEFAULTS: ClassVar[Mapping[str, Any]] = {}

    #: Settings that no longer do anything, each with what to do instead. A
    #: run that still passes one is warned rather than refused. Merged down
    #: the class hierarchy like :attr:`DEFAULTS`.
    RETIRED: ClassVar[Mapping[str, str]] = {}

    #: Settings that are gone and cannot be ignored, each with what to write
    #: instead. A run passing one is refused, because carrying on would drive
    #: the robot differently from what the setting asked for. Merged down the
    #: class hierarchy like :attr:`DEFAULTS`.
    REFUSED: ClassVar[Mapping[str, str]] = {}

    #: Settings the preset reads itself, as a dataclass: options it passes to
    #: the robot, such as a controller mode, and how it builds its channels.
    #: ``None`` when it reads none.
    OPTIONS: ClassVar[Optional[type]] = None

    #: Whether this id may stand for its task's robot-free id, such as
    #: ``PegInsertion-v1``. ``False`` on an id that differs from another on the
    #: same robot in how the arms are driven, which a run must choose itself
    #: because its checkpoint was trained against one action layout.
    GENERIC_ID: ClassVar[bool] = True

    def __init__(
        self,
        override_cfg: Optional[Mapping[str, Any]] = None,
        worker_info: Optional[WorkerInfo] = None,
        robot_info: "Optional[RobotInfo[Any]]" = None,
        env_idx: int = 0,
    ) -> None:
        cls = type(self)
        owners = [RegisteredTaskEnvConfig, cls.ACTION_CONFIG, cls.TASK.CONFIG]
        if cls.OPTIONS is not None:
            owners.append(cls.OPTIONS)
        # A preset's default for a setting its task does not have is dropped,
        # so a robot's defaults never stop it running a task that lacks one.
        declared = {
            field.name for owner in owners for field in dataclasses.fields(owner)
        }
        settings = {
            key: value for key, value in cls.defaults().items() if key in declared
        }
        settings.update(override_cfg or {})
        refused = [key for key in cls._merged("REFUSED") if key in settings]
        if refused:
            instead = cls._merged("REFUSED")
            lines = "\n".join(f"  {key!r}: {instead[key]}" for key in sorted(refused))
            raise ValueError(
                f"{cls.__name__} no longer takes these settings, and ignoring "
                f"them would drive the robot differently:\n{lines}"
            )
        for key, instead in cls._merged("RETIRED").items():
            if key in settings:
                settings.pop(key)
                warnings.warn(
                    f"{cls.__name__}: {key!r} is retired and ignored. {instead}",
                    DeprecationWarning,
                    stacklevel=2,
                )
        config, action_config, task_config, *options = split_overrides(
            settings, owners, owner=cls.__name__
        )
        self.options = options[0] if options else None
        self.robot_info = robot_info
        self.env_idx = env_idx
        self.hardware = get_hardware_config(
            cls.robot_config(), robot_info, is_dummy=config.is_dummy
        )
        # Everything a config can get wrong is checked before the robot is
        # composed, so a bad run never opens hardware.
        task = cls.TASK(task_config)
        # The task names the roles, so a preset builds one arm's channels or
        # two from the same code.
        roles = tuple(task.requirements())
        action = cls.make_action(self.hardware, action_config, self.options, roles)
        cameras = tuple(cls.camera_infos(self.hardware, config))
        observation = cls.make_observation(self.hardware, cameras, roles)
        if len(cameras) < observation.min_cameras:
            raise ValueError(
                f"{cls.__name__} requires robot_info with at least "
                f"{observation.min_cameras} camera serial(s), including in "
                "dummy mode."
            )

        robot, reward_model = None, None
        node_rank = worker_info.cluster_node_rank if worker_info else 0
        worker_rank = worker_info.rank if worker_info else 0
        if not config.is_dummy:
            robot = cls.ROBOT.from_config(
                self.hardware,
                cameras={camera.name: camera for camera in cameras},
                env_idx=env_idx,
                node_rank=node_rank,
                worker_rank=worker_rank,
                **cls.robot_options(self.options),
            )
            if config.use_reward_model:
                reward_model = RewardModel.launch(
                    config.reward_worker_cfg,
                    image_key=config.reward_image_key,
                    node_rank=node_rank
                    if config.reward_worker_node_rank is None
                    else config.reward_worker_node_rank,
                    node_group=config.reward_worker_node_group,
                    hardware_rank=config.reward_worker_hardware_rank,
                    env_idx=env_idx,
                    worker_rank=worker_rank,
                )
        super().__init__(
            robot,
            task,
            action,
            observation=observation,
            config=config,
            reward_model=reward_model,
        )

    @classmethod
    def defaults(cls) -> dict[str, Any]:
        """Defaults merged from the robot preset down to this task id."""
        return cls._merged("DEFAULTS")

    @classmethod
    def _merged(cls, name: str) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        for klass in reversed(cls.__mro__):
            merged.update(vars(klass).get(name, {}))
        return merged

    @classmethod
    def robot_config(cls) -> type[RobotConfig]:
        """The hardware config class registered for :attr:`ROBOT`."""
        return RobotDiscovery.registry[cls.ROBOT.ROBOT_TYPE].config_cls

    @classmethod
    def camera_infos(
        cls, hardware: RobotConfig, config: RegisteredTaskEnvConfig
    ) -> Iterable[CameraInfo]:
        """Declare the hardware's cameras as the run names and crops them.

        A camera is ``wrist_1``, ``wrist_2``, and so on in the hardware's
        order unless ``camera_names`` names its serial.
        """
        camera_type = getattr(hardware, "camera_type", None) or "realsense"
        serials = getattr(hardware, "camera_serials", None) or []
        names = config.camera_names or {}
        regions = config.camera_crop_regions or {}
        for index, serial in enumerate(map(str, serials), start=1):
            name = names.get(serial, f"wrist_{index}")
            region = regions.get(serial)
            yield CameraInfo(
                name=name,
                serial_number=serial,
                camera_type=camera_type,
                crop_region=None
                if region is None
                else crop_region(region, camera=name, serial=serial),
                enable_depth=config.enable_camera_depth,
            )

    @classmethod
    def make_action(
        cls,
        hardware: RobotConfig,
        config: Any,
        options: Any = None,
        roles: tuple[str, ...] = ("arm",),
    ) -> ActionLayout:
        """The channels this robot's action is cut into.

        Args:
            hardware: The robot's hardware config.
            config: The run's action settings, an :attr:`ACTION_CONFIG`.
            options: The preset's own settings, an :attr:`OPTIONS`, or ``None``.
            roles: The roles the task asks for, in the order it named them. A
                robot with one arm has one role and can ignore this; one whose
                task chooses how many arms to drive builds a channel per role.
        """
        raise NotImplementedError(f"{cls.__name__} does not define make_action().")

    @classmethod
    def robot_options(cls, options: Any) -> Mapping[str, Any]:
        """Keyword options for ``ROBOT.from_config``, from :attr:`OPTIONS`."""
        return {}

    @classmethod
    def make_observation(
        cls,
        hardware: RobotConfig,
        cameras: tuple[CameraInfo, ...],
        roles: tuple[str, ...] = ("arm",),
    ) -> ObservationSpec:
        """What this robot's policy observes, for the roles the task asks for."""
        raise NotImplementedError(f"{cls.__name__} does not define make_observation().")
