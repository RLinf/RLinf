import copy
import logging
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    MotorChain,
    MotorInfo,
)
from i2rt.robots.robot import Robot
from i2rt.robots.utils import GripperForceLimiter, GripperType, JointMapper
from i2rt.utils.mujoco_utils import MuJoCoKDL


@dataclass
class JointStates:
    names: List[str]
    pos: np.ndarray
    vel: np.ndarray
    eff: np.ndarray
    temp_mos: np.ndarray  # MOS temperature (float): Motor MOS temperature.
    temp_rotor: np.ndarray  # ROTOR temperature (float): Motor ROTOR temperature.

    def asdict(self) -> Dict[str, Any]:
        return {
            "names": self.names,
            "pos": self.pos.flatten().tolist(),
            "vel": self.vel.flatten().tolist(),
            "eff": self.eff.flatten().tolist(),
        }


@dataclass
class JointCommands:
    torques: np.ndarray

    pos: np.ndarray
    vel: np.ndarray
    kp: np.ndarray
    kd: np.ndarray

    indices: Optional[List[int]] = None

    @classmethod
    def init_all_zero(cls, n_joints: int) -> "JointCommands":
        return cls(
            torques=np.zeros(n_joints),
            pos=np.zeros(n_joints),
            vel=np.zeros(n_joints),
            kp=np.zeros(n_joints),
            kd=np.zeros(n_joints),
        )


class MotorChainRobot(Robot):
    """A generic Robot protocol."""

    def __init__(
        self,
        motor_chain: MotorChain,
        xml_path: Optional[str] = None,
        use_gravity_comp: bool = True,
        gravity: Optional[np.ndarray] = None,
        gravity_comp_factor: float = 1.0,  # New parameter with default value
        gripper_index: Optional[int] = None,  # Zero starting index: if you have a 6 dof arm and last one is gripper: 6
        kp: Union[float, List[float]] = 10.0,
        kd: Union[float, List[float]] = 1.0,
        joint_limits: Optional[np.ndarray] = None,  # if provided, override the mujoco xml joint limits
        gripper_limits: Optional[np.ndarray] = None,  # [closed, open]
        limit_gripper_force: float = -1,  # whether to limit the gripper effort when it is blocked. -1 means no limit.
        clip_motor_torque: float = np.inf,  # clip the offset motor torque, real motor torque can still still be larger than this setting depending on the motor onboard PID loop
        gripper_type: GripperType = GripperType.CRANK_4310,
        temp_record_flag: bool = False,  # whether record the motor's temperature
        enable_gripper_calibration: bool = False,  # whether to auto-detect gripper limits
        zero_gravity_mode:bool = True,
        # below are calibration parameters
        test_torque: float = 0.5,  # test torque for gripper detection (Nm)
        test_duration: float = 2.0,  # max test duration for each direction (s)
        position_threshold: float = 0.01,  # minimum position change to consider motor still moving (rad)
        check_interval: float = 0.05,  # time interval between checks (s)
    ) -> None:
        self.temp_record_flag = temp_record_flag
        if gripper_index is not None:
            assert gripper_index == len(motor_chain) - 1, (
                "Gripper index should be the last one, but got {gripper_index}"
            )

            # Auto-detect gripper limits if enabled and gripper_limits is None
            print(
                f"initializing motorchain robot, gripper_limits: {gripper_limits}, enable_gripper_calibration: {enable_gripper_calibration}"
            )
            if gripper_limits is None and enable_gripper_calibration:
                from i2rt.robots.utils import detect_gripper_limits

                logger = logging.getLogger(__name__)
                logger.info("Auto-detecting gripper limits...")
                detected_limits = detect_gripper_limits(
                    motor_chain=motor_chain,
                    gripper_index=gripper_index,
                    test_torque=test_torque,
                    max_duration=test_duration,
                    position_threshold=position_threshold,
                    check_interval=check_interval,
                )
                gripper_limits = np.array(detected_limits)
                logger.info(f"Gripper limits auto-detected: {gripper_limits}")
            elif gripper_limits is None:
                raise ValueError(
                    f"{self}: Gripper limits are required if gripper index is provided and auto-calibration is disabled."
                )
            else:
                # Use the provided gripper_limits
                logger = logging.getLogger(__name__)
                logger.info(f"Using provided gripper limits: {gripper_limits}")


        self._last_gripper_command_qpos = 1 # initialize as fully open
        assert clip_motor_torque >= 0.0
        self._clip_motor_torque = clip_motor_torque
        self.motor_chain = motor_chain
        self.use_gravity_comp = use_gravity_comp
        self.gravity_comp_factor = gravity_comp_factor  # Store the factor

        # variables for gripper effort limiting
        self._gripper_index = gripper_index
        self.remapper = JointMapper({}, len(motor_chain))  # so it works without gripper
        self._gripper_limits = gripper_limits

        if self._gripper_index is not None:
            self._gripper_force_limiter = GripperForceLimiter(
                max_force=limit_gripper_force, gripper_type=gripper_type, kp=kp[gripper_index]
            )  # force in newton
            self._limit_gripper_force = limit_gripper_force

            self.remapper = JointMapper(
                index_range_map={gripper_index: gripper_limits},
                total_dofs=len(motor_chain),
            )

        # make sure kp, kd are float number not int
        self._kp = (
            np.array(
                [
                    kp,
                ]
                * len(motor_chain)
            )
            if isinstance(kp, float)
            else np.array(kp)
        )
        self._kd = (
            np.array(
                [
                    kd,
                ]
                * len(motor_chain)
            )
            if isinstance(kd, float)
            else np.array(kd)
        )

        self._joint_limits:Optional[np.ndarray] = None
        if xml_path is not None:
            self.xml_path = os.path.expanduser(xml_path)
            self.kdl = MuJoCoKDL(self.xml_path)
            if gravity is not None:
                self.kdl.set_gravity(gravity)
            # Load the joint limits from the xml file
            self._joint_limits = self.kdl.joint_limits
        else:
            assert use_gravity_comp is False, "Gravity compensation requires a valid XML path."

        # override the xml joint limits with the provided joint_limits
        if joint_limits is not None:
            joint_limits = np.array(joint_limits)
            assert np.all(joint_limits[:, 0] < joint_limits[:, 1]), (
                "Lower joint limits must be smaller than upper limits"
            )
            self._joint_limits = joint_limits
        self._command_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._joint_state: Optional[JointStates] = None
        while self._joint_state is None:
            # wait to recive joint data
            time.sleep(0.05)
            self._joint_state = self._motor_state_to_joint_state(self.motor_chain.read_states())
        self._commands = JointCommands.init_all_zero(len(motor_chain))
        # For SWE-454, check if the current qpos is in the joint limits
        self._check_current_qpos_in_joint_limits()

        self._stop_event = threading.Event()  # Add a stop event
        self._server_thread = threading.Thread(target=self.start_server, name="robot_server")
        self._server_thread.start()

        if not zero_gravity_mode:
            # set current qpos as target pos with the default PD parameters
            self.command_joint_pos(self._joint_state.pos)
    def __repr__(self) -> str:
        return f"MotorChainRobot(motor_chain={self.motor_chain})"

    def _check_current_qpos_in_joint_limits(self, buffer_rad: float = 0.1) -> None:
        """Check if the self._joint_state is in the joint limits.
        If violated, raise an error.
        """
        if self._joint_state is None or self._joint_limits is None:
            raise RuntimeError(f"{self}: Joint limits:{self._joint_limits} or joint state:{self._joint_state} are not set.")

        current_pos = self._joint_state.pos

        # Check arm joints (exclude gripper if present)
        if self._gripper_index is not None:
            # Only check arm joints, not the gripper
            arm_pos = current_pos[:self._gripper_index]
            arm_limits = self._joint_limits
        else:
            # Check all joints
            arm_pos = current_pos
            arm_limits = self._joint_limits

        # Check if any joint is outside its limits
        lower_limits = arm_limits[:, 0] - buffer_rad
        upper_limits = arm_limits[:, 1] + buffer_rad


        # Find joints that violate lower limits
        lower_violations = arm_pos < lower_limits
        # Find joints that violate upper limits
        upper_violations = arm_pos > upper_limits

        if np.any(lower_violations) or np.any(upper_violations):
            violation_details = []

            for i, (pos, lower, upper) in enumerate(zip(arm_pos, lower_limits, upper_limits)):
                if pos < lower:
                    violation_details.append(f"Joint {i}: {pos:.4f} < {lower:.4f} (lower limit)")
                elif pos > upper:
                    violation_details.append(f"Joint {i}: {pos:.4f} > {upper:.4f} (upper limit)")

            violation_msg = "; ".join(violation_details)
            # turn off the main motor control thread as well.
            self.motor_chain.running = False
            raise RuntimeError(f"{self}: Joint limit violation detected: {violation_msg}, the root reason should be zero position offset. possible solution: 1. move the arm to zero position and power cycle the robot. 2. Recalibrate the motor zero position.")

    def get_robot_info(self) -> Dict[str, Any]:
        """Get the robot information, such as kp, kd, joint limits, gripper limits, etc."""
        return {
            "kp": self._kp,
            "kd": self._kd,
            "joint_limits": self._joint_limits,
            "gripper_limits": self._gripper_limits,
            "gravity_comp_factor": self.gravity_comp_factor,
            "limit_gripper_effort": self._limit_gripper_force,
            "gripper_index": self._gripper_index,
        }

    def start_server(self) -> None:
        """Start the server."""
        last_time = time.time()
        iteration_count = 0
        self.update()

        logging.info("initializing, ....")

        while not self._stop_event.is_set():  # Check the stop event
            current_time = time.time()
            elapsed_time = current_time - last_time

            self.update()
            if not self.motor_chain.running:
                raise RuntimeError(f"{self}: motor_chain_robot's motor chain is not running, exiting the robot server")
            time.sleep(0.004)

            iteration_count += 1
            if elapsed_time >= 10.0:
                control_frequency = iteration_count / elapsed_time
                # Overwrite the current line with the new frequency information
                logging.info(f"{self}: Grav Comp Control Frequency: {control_frequency:.2f} Hz")
                if control_frequency < 100:
                    logging.warning(
                        f"{self}: Gravity compensation control loop is slow, current frequency: {control_frequency:.2f} Hz"
                    )
                # Reset the counter and timer
                last_time = current_time
                iteration_count = 0

    def update(self) -> None:
        """Update the robot.

        Send Torques and update the joint state.
        """
        with self._command_lock:
            joint_commands = copy.deepcopy(self._commands)
        with self._state_lock:
            g = self._compute_gravity_compensation(self._joint_state)
            motor_torques = joint_commands.torques + g * self.gravity_comp_factor
            motor_torques = np.clip(motor_torques, -self._clip_motor_torque, self._clip_motor_torque)

            if self._gripper_index is not None:
                if self._limit_gripper_force > 0 and self._joint_state is not None:
                    # Get current gripper state in raw robot joint pos space
                    gripper_state = {
                        "target_qpos": joint_commands.pos[self._gripper_index],
                        "current_qpos": self.remapper.to_robot_joint_pos_space(self._joint_state.pos)[
                            self._gripper_index
                        ],
                        "current_qvel": self._joint_state.vel[self._gripper_index],
                        "current_eff": self._joint_state.eff[self._gripper_index],
                        "current_normalized_qpos": self._joint_state.pos[self._gripper_index],
                        "target_normalized_qpos": self.remapper.to_command_joint_pos_space(joint_commands.pos)[
                            self._gripper_index
                        ],
                        "last_command_qpos": self._last_gripper_command_qpos,
                    }

                    joint_commands.pos[self._gripper_index] = self._gripper_force_limiter.update(gripper_state)

                # add final clip so the gripper won't be over-adjusted
                joint_commands.pos[self._gripper_index] = np.clip(
                    joint_commands.pos[self._gripper_index],
                    min(self._gripper_limits),
                    max(self._gripper_limits),
                )
                self._last_gripper_command_qpos = joint_commands.pos[self._gripper_index]
            if not self.motor_chain.start_thread_flag:
                self.motor_chain.set_commands(
                    motor_torques,
                    pos=joint_commands.pos,
                    vel=joint_commands.vel,
                    kp=joint_commands.kp,
                    kd=joint_commands.kd,
                )
                self.motor_chain.start_thread()
                self.motor_chain.start_thread_flag = True
            # Send commands to motor chain and update joint state
            motor_state = self.motor_chain.set_commands(
                motor_torques,
                pos=joint_commands.pos,
                vel=joint_commands.vel,
                kp=joint_commands.kp,
                kd=joint_commands.kd,
            )
            self._joint_state = self._motor_state_to_joint_state(motor_state)
            # For SWE-454, check if the current qpos is in the joint limits
            # When the arm is fully extened and got a power cycle, the initial qpos might still with the range, then we need to keep monitoring the qpos during the robot running.
            self._check_current_qpos_in_joint_limits()

    def _motor_state_to_joint_state(self, motor_state: List[MotorInfo]) -> JointStates:
        """Convert motor state to joint state.

        Args:
            motor_state (List[Any]): The motor state.

        Returns:
            Dict[str, np.ndarray]: The joint state.
        """
        names = [str(i) for i in range(len(motor_state))]
        pos = np.array([motor.pos for motor in motor_state])
        pos = self.remapper.to_command_joint_pos_space(pos)
        vel = np.array([motor.vel for motor in motor_state])
        vel = self.remapper.to_command_joint_vel_space(vel)
        eff = np.array([motor.eff for motor in motor_state])
        temp_mos = np.array([motor.temp_mos for motor in motor_state])
        temp_rotor = np.array([motor.temp_rotor for motor in motor_state])
        return JointStates(
            names=names,
            pos=pos,
            vel=vel,
            eff=eff,
            temp_mos=temp_mos,
            temp_rotor=temp_rotor,
        )

    def _compute_gravity_compensation(self, joint_state: Optional[Dict[str, np.ndarray]]) -> np.ndarray:
        if joint_state is None or not self.use_gravity_comp:
            return np.zeros(len(self.motor_chain))
        elif self.use_gravity_comp:
            q = joint_state.pos[: self._gripper_index] if self._gripper_index is not None else joint_state.pos
            t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            # print gravity torque to 2f
            if np.max(np.abs(t)) > 20.0:
                print([f"{s:.2f}" for s in t])
                raise RuntimeError(f"{self}: too large torques")
            if self._gripper_index is None:
                return self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
            else:
                t = self.kdl.compute_inverse_dynamics(q, np.zeros(q.shape), np.zeros(q.shape))
                return np.append(t, 0.0)

    # ----------------- Server Functions ----------------- #

    def num_dofs(self) -> int:
        """Get the number of joints of the robot, including the gripper.

        Returns:
            int: The number of joints of the robot.
        """
        return len(self.motor_chain)

    def get_joint_pos(self) -> np.ndarray:
        """Get the current state of the leader robot, including the gripper in radian.

        Returns:
            T: The current state of the leader robot.
        """
        with self._state_lock:
            return self._joint_state.pos

    def _clip_robot_joint_pos_command(self, pos: np.ndarray) -> np.ndarray:
        """Clip the robot joint pos command to the joint limits. Do not clip the gripper pos.
        Args:
            pos (np.ndarray): The joint pos command to clip.
        Returns:
            np.ndarray: The clipped joint pos command.
        """

        if self._joint_limits is not None:
            if self._gripper_index is not None:
                pos[: self._gripper_index] = np.clip(
                    pos[: self._gripper_index],
                    self._joint_limits[:, 0],
                    self._joint_limits[:, 1],
                )
            else:
                pos = np.clip(pos, self._joint_limits[:, 0], self._joint_limits[:, 1])
        return pos

    def command_joint_pos(self, joint_pos: np.ndarray) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_pos (np.ndarray): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_pos)
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.kp = self._kp
            self._commands.kd = self._kd

    def command_joint_state(self, joint_state: Dict[str, np.ndarray]) -> None:
        """Command the leader robot to a given state.

        Args:
            joint_state (Dict[str, np.ndarray]): The state to command the leader robot to.
        """
        pos = self._clip_robot_joint_pos_command(joint_state["pos"])
        vel = joint_state["vel"]
        self._commands = JointCommands.init_all_zero(len(self.motor_chain))
        kp = joint_state.get("kp", self._kp)
        kd = joint_state.get("kd", self._kd)
        with self._command_lock:
            self._commands.pos = self.remapper.to_robot_joint_pos_space(pos)
            self._commands.vel = self.remapper.to_robot_joint_vel_space(vel)
            self._commands.kp = kp
            self._commands.kd = kd

    def zero_torque_mode(self) -> None:
        logging.info(f"Entering zero_torque_mode for {self}")
        with self._command_lock:
            self._commands = JointCommands.init_all_zero(len(self.motor_chain))
            self._kp = np.zeros(len(self.motor_chain))
            self._kd = np.zeros(len(self.motor_chain))

    def get_observations(self) -> Dict[str, np.ndarray]:
        """Get the current observations of the robot.

        This is to extract all the information that is available from the robot,
        such as joint positions, joint velocities, etc. This may also include
        information from additional sensors, such as cameras, force sensors, etc.

        Returns:
            Dict[str, np.ndarray]: A dictionary of observations.
        """
        with self._state_lock:
            if self._gripper_index is None:
                result = {
                    "joint_pos": self._joint_state.pos,
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            else:
                result = {
                    "joint_pos": self._joint_state.pos[: self._gripper_index],
                    "gripper_pos": np.array([self._joint_state.pos[self._gripper_index]]),
                    "joint_vel": self._joint_state.vel,
                    "joint_eff": self._joint_state.eff,
                }
            if self.temp_record_flag:
                result["temp_mos"] = self._joint_state.temp_mos
                result["temp_rotor"] = self._joint_state.temp_rotor
            return result

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Exit the runtime context related to this object."""
        self.close()

    def move_joints(self, target_joint_positions: np.ndarray, time_interval_s: float = 2.0) -> None:
        """Move the robot to a given joint positions."""
        with self._state_lock:
            current_pos = self._joint_state.pos
        assert len(current_pos) == len(target_joint_positions)
        steps = 50  # 50 steps over time_interval_s
        for i in range(steps + 1):
            alpha = i / steps  # Interpolation factor
            target_pos = (1 - alpha) * current_pos + alpha * target_joint_positions  # Linear interpolation
            self.command_joint_pos(target_pos)
            time.sleep(time_interval_s / steps)

    def close(self) -> None:
        """Safely close the robot by setting all torques to zero."""
        # self.move_to_zero()
        self._stop_event.set()  # Signal the thread to stop
        self._server_thread.join()  # Wait for the thread to finish
        self.motor_chain.close()
        print("Robot closed with all torques set to zero.")

    def update_kp_kd(self, kp: np.ndarray, kd: np.ndarray) -> None:
        assert kp.shape == self._kp.shape == kd.shape
        self._kp = kp
        self._kd = kd


if __name__ == "__main__":
    import argparse
    import time

    from i2rt.robots.get_robot import get_yam_robot
    from i2rt.utils.utils import override_log_level

    override_log_level(level=logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--gripper_type", type=str, default="crank_4310")
    args.add_argument("--channel", type=str, default="can0")
    args.add_argument("--operation_mode", type=str, default="gravity_comp")

    args = args.parse_args()

    gripper_type = GripperType.from_string_name(args.gripper_type)

    print(f"Initializing yam with gripper type: {gripper_type}")
    robot = get_yam_robot(args.channel, gripper_type=gripper_type)

    if args.operation_mode == "gravity_comp":
        while True:
            # print(robot.get_observations())
            time.sleep(1)
    elif args.operation_mode == "test_gripper":
        assert gripper_type != GripperType.YAM_TEACHING_HANDLE, (
            "test_gripper is not supported for YAM_TEACHING_HANDLE, teaching handle is a passive device"
        )
        for _ in range(30):
            for gripper_pos in [0.8, 0.0]:
                print(f"gripper_pos: {gripper_pos}")
                robot.command_joint_pos(np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, gripper_pos]))
                time.sleep(4)
                print(robot.get_observations())
    elif args.operation_mode == "stay_current_qpos":
        current_qpos = robot.get_joint_pos()
        robot.command_joint_pos(current_qpos)
        while True:
            time.sleep(1)
