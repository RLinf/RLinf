import enum
import logging
import os
import struct
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Protocol, Tuple
from i2rt.motor_drivers.utils import MotorInfo, MotorConstants, FeedbackFrameInfo, MotorErrorCode, MotorType, uint_to_float, float_to_uint, MotorInfo, ReceiveMode
import can
import numpy as np
from i2rt.motor_drivers.can_interface import CanInterface

from i2rt.utils.utils import RateRecorder

log_level = os.getenv("LOGLEVEL", "ERROR").upper()

# if no log_level is set, set it to WARNING
logging.basicConfig(level=logging.ERROR)

# set control frequence
CONTROL_FREQ = 250
CONTROL_PERIOD = 1.0 / CONTROL_FREQ  # 4 ms

EXPECTED_CONTROL_PERIOD = 0.007
REPORT_INTERVAL = 30.0


class ControlMode:
    MIT = "MIT"
    POS_VEL = "POS_VEL"
    VEL = "VEL"

    @classmethod
    def get_id_offset(cls, control_mode: str) -> int:
        if control_mode == cls.MIT:
            return 0x000
        elif control_mode == cls.POS_VEL:
            return 0x100
        elif control_mode == cls.VEL:
            return 0x200
        else:
            raise ValueError(f"Control mode '{control_mode}' not recognized.")


######### for passive encoder #########
@dataclass
class PassiveEncoderInfo:
    """The encoder report."""

    id: int
    """The device number, uint8."""
    position: float
    """Position, in radian."""
    velocity: float
    """Velocity, in radian/s."""
    io_inputs: List[bool]
    """The discrete inputs, list of boolean."""


class PassiveEncoderReader:
    def __init__(self, can_interface: CanInterface, receive_mode: ReceiveMode = ReceiveMode.plus_one):
        self.can_interface = can_interface
        # assert self.can_interface.use_buffered_reader, "Passive encoder reader must use buffered reader"

        self.receive_mode = receive_mode

    def read_encoder(self, encoder_id: int) -> PassiveEncoderInfo:
        # this encoder's trigger message is 0x02
        data = [0xFF, 0x02]
        message = self.can_interface._send_message_get_response(
            encoder_id, encoder_id, data, expected_id=self.receive_mode.get_receive_id(0x50E), max_retry=15
        )
        pos, vel, button_state = self._parse_encoder_message(message)
        pos_range = [-0.7, 0.7]
        pos = np.clip(pos, pos_range[0], pos_range[1])
        # normalize pos to 1 - 0
        delta = np.abs(0.0 - pos)
        pos = delta / 0.7
        result = PassiveEncoderInfo(id=encoder_id, position=pos, velocity=vel, io_inputs=button_state)
        return result

    def _parse_encoder_message(self, message: can.Message) -> PassiveEncoderInfo:
        # Standard format
        struct_format = "!B h h B"
        device_id, position, velocity, digital_inputs = struct.unpack(struct_format, message.data)

        # Convert position and velocity to radians
        position_rad = position * 2 * np.pi / 4096
        velocity_rad = velocity * 2 * np.pi / 4096
        button_state = [digital_inputs % 2, digital_inputs // 2]

        return position_rad, velocity_rad, button_state


class EncoderChain:
    def __init__(self, encoder_ids: List[int], encoder_interface: CanInterface):
        self.encoder_ids = encoder_ids
        self.encoder_interface = encoder_interface

    def read_states(self) -> List[PassiveEncoderInfo]:
        return [self.encoder_interface.read_encoder(encoder_id) for encoder_id in self.encoder_ids]


class DMSingleMotorCanInterface(CanInterface):
    """Class for CAN interface with a single motor."""

    def __init__(
        self,
        control_mode: str = ControlMode.MIT,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "socketcan",
        bitrate: int = 1000000,
        receive_mode: ReceiveMode = ReceiveMode.p16,
        name: str = "default_can_DM_interface",
        use_buffered_reader: bool = False,
    ):
        super().__init__(
            channel, bustype, bitrate, receive_mode=receive_mode, name=name, use_buffered_reader=use_buffered_reader
        )
        self.control_mode = control_mode
        self.cmd_idoffset = ControlMode.get_id_offset(self.control_mode)
        self.receive_mode = receive_mode

    def _get_frame_id(self, motor_id: int) -> int:
        """Calculate the Control Frame ID for a given motor."""
        return self.cmd_idoffset + motor_id

    def motor_on(self, motor_id: int, motor_type: str) -> None:
        """Turn on the motor.

        Args:
            motor_id (int): The ID of the motor to turn on.
        """
        current_level = logging.getLogger().getEffectiveLevel()
        logging.getLogger().setLevel(logging.ERROR)

        id = motor_id  # self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFC]
        
        message = self._send_message_get_response(id, motor_id, data)

        # dummy motor type just check motor status
        motor_info = self.parse_recv_message(message, MotorType.DM4310, ignore_error=True)
        if int(motor_info.error_code, 16) != MotorErrorCode.normal:
            while int(motor_info.error_code, 16) != MotorErrorCode.normal:
                logging.info(f"motor {motor_id} error: {motor_info.error_message}")
                self.clean_error(motor_id=motor_id)
                self.try_receive_message()
                logging.info(f"motor {motor_id} error cleaned")
                # enable again

                message = self._send_message_get_response(id, motor_id, data)
                motor_info = self.parse_recv_message(message, motor_type, ignore_error=True)
        else:
            logging.info(f"motor {motor_id} is already on")
        logging.getLogger().setLevel(current_level)
        motor_info = self.parse_recv_message(message, motor_type)
        return motor_info

    def clean_error(self, motor_id: int) -> None:
        # self.try_receive_message()
        id = motor_id  # self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFB]
        logging.info("clear error")
        message = can.Message(arbitration_id=motor_id, data=data, is_extended_id=False)
        for _ in range(3):
            try:
                self.bus.send(message)
            except Exception as e:
                logging.warning(e)
                logging.warning(
                    "\033[91m" + "CAN Error: Failed to communicate with motor over can bus. Retrying..." + "\033[0m"
                )
        # message = self._send_message_get_response(id, data)

    def motor_off(self, motor_id: int) -> None:
        """Turn off the motor.

        Args:
            motor_id (int): The ID of the motor to turn off.
        """
        id = self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFD]
        message = self._send_message_get_response(id, motor_id, data)

    def save_zero_position(self, motor_id: int) -> None:
        """Save the current position as zero position.

        Args:
            motor_id (int): The ID of the motor to save zero position.
        """
        id = self._get_frame_id(motor_id)
        data = [0xFF] * 7 + [0xFE]
        try:
            message = self._send_message_get_response(id, motor_id, data, 2)
        except AssertionError:
            pass
        # check if set zero position success
        current_state = self.set_control(id, MotorType.DM4310, 0, 0, 0, 0, 0)
        diff = abs(current_state.position)
        if diff < 0.01:
            logging.info(f"motor {motor_id} set zero position success, current position: {current_state.position}")
        # message = self._receive_message(timeout=0.5)

    def set_control(
        self,
        motor_id: int,
        motor_type: str,
        pos: float,
        vel: float,
        kp: float,
        kd: float,
        torque: float,
    ) -> FeedbackFrameInfo:
        """Set the control of the motor and return its status.

        Args:
            motor_id (int): The ID of the motor. Check GUI for the CAN ID.
            motor_type (str): The type of the motor. Check MotorType class for available motor types.
            pos (float): The target position value.
            vel (float): The target velocity value.
            kp (float): The proportional gain value.
            kd (float): The derivative gain value.
            torque (float): The target torque value.

        Returns:
            FeedbackFrameInfo: The current state of the motor, including motor id, error code, position, velocity, torque, temperature.
        """
        frame_id = self._get_frame_id(motor_id)
        # Prepare the CAN message
        data = bytearray(8)
        if self.control_mode == ControlMode.MIT:
            const = MotorType.get_motor_constants(motor_type)

            pos_tmp = float_to_uint(pos, const.POSITION_MIN, const.POSITION_MAX, 16)
            vel_tmp = float_to_uint(vel, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
            kp_tmp = float_to_uint(kp, const.KP_MIN, const.KP_MAX, 12)
            kd_tmp = float_to_uint(kd, const.KD_MIN, const.KD_MAX, 12)
            tor_tmp = float_to_uint(torque, const.TORQUE_MIN, const.TORQUE_MAX, 12)

            # "& 0xFF" (bitwise AND with 0xFF) is used to ensure that only the lowest 8 bits (one byte) of a value are kept,
            # and any higher bits are discarded.
            data[0] = (pos_tmp >> 8) & 0xFF
            data[1] = pos_tmp & 0xFF
            data[2] = (vel_tmp >> 4) & 0xFF
            data[3] = ((vel_tmp & 0xF) << 4) | (kp_tmp >> 8)
            data[4] = kp_tmp & 0xFF
            data[5] = (kd_tmp >> 4) & 0xFF
            data[6] = ((kd_tmp & 0xF) << 4) | (tor_tmp >> 8)
            data[7] = tor_tmp & 0xFF
        elif self.control_mode == ControlMode.VEL:
            # system will only response to vel command
            can_data = struct.pack("<f", vel)
            data[0:4] = can_data[0:4]

        # Send the CAN message
        message = self._send_message_get_response(frame_id, motor_id, data, max_retry=15)

        # Parse the received message to extract motor information
        motor_info = self.parse_recv_message(message, motor_type)
        return motor_info

    def parse_recv_message(
        self, message: can.Message, motor_type: str, ignore_error: bool = False
    ) -> FeedbackFrameInfo:
        """Parse the received message to extract motor information.

        Args:
            message (can.Message): The received message.

        Returns:
            FeedbackFrameInfo: The current state of the motor.
        """
        data = message.data
        error_int = (data[0] & 0xF0) >> 4  # TODO: error code seems incorrect, double check

        # convert error into hex
        error_hex = hex(error_int)
        error_message = MotorErrorCode.get_error_message(error_int)

        motor_id_of_this_response = self.receive_mode.to_motor_id(message.arbitration_id)
        if error_hex != "0x1":
            logging.warning(
                f"motor id: {motor_id_of_this_response}, error: {error_message} at {self.name} and channel {self.bus.channel_info}"
            )
            if not ignore_error:
                logging.error(
                    f"motor id: {motor_id_of_this_response}, error: {error_message} at {self.name} and channel {self.bus.channel_info}"
                )
                raise RuntimeError(
                    f"Motor error detected: motor id: {motor_id_of_this_response}, error: {error_message}"
                )
        p_int = (data[1] << 8) | data[2]
        v_int = (data[3] << 4) | (data[4] >> 4)
        t_int = ((data[4] & 0xF) << 8) | data[5]
        temporature_mos = data[6]
        temperature_rotor = data[7]

        const = MotorType.get_motor_constants(motor_type)
        position = uint_to_float(p_int, const.POSITION_MIN, const.POSITION_MAX, 16)
        velocity = uint_to_float(v_int, const.VELOCITY_MIN, const.VELOCITY_MAX, 12)
        torque = uint_to_float(t_int, const.TORQUE_MIN, const.TORQUE_MAX, 12)
        temperature_mos = float(temporature_mos)
        temperature_rotor = float(temperature_rotor)

        return FeedbackFrameInfo(
            id=motor_id_of_this_response,
            error_code=error_hex,
            error_message=error_message,
            position=position,
            velocity=velocity,
            torque=torque,
            temperature_mos=temperature_mos,
            temperature_rotor=temperature_rotor,
        )


@dataclass
class MotorCmd:
    type: str = "pos_vel_torque"
    pos: float = 0.0
    vel: float = 0.0
    torque: float = 0.0
    kp: float = 0.0
    kd: float = 0.0


class MotorChain(Protocol):
    """Class for CAN interface with multiple motors."""

    def __len__(self) -> int:
        """Get the number of motors in the chain."""
        raise NotImplementedError

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        """Set commands to the motors in the chain."""
        raise NotImplementedError


class DMChainCanInterface(MotorChain):
    def __init__(
        self,
        motor_list: List[Tuple[int, str]],
        motor_offset: np.ndarray,
        motor_direction: np.ndarray,
        channel: str = "PCAN_USBBUS1",
        bitrate: int = 1000000,
        start_thread: bool = True,  # If true, will start the internal motor reading loop
        motor_chain_name: str = "default_motor_chain",
        receive_mode: ReceiveMode = ReceiveMode.p16,
        control_mode: ControlMode = ControlMode.MIT,
        get_same_bus_device_driver: Optional[Callable] = None,
        use_buffered_reader: bool = False,  # buffered reader is not very stable, the latest encoder fix allows us to use the non-buffered reader
    ):
        assert not use_buffered_reader, (
            "buffered reader is not very stable, the latest encoder fix allows us to use the non-buffered reader"
        )
        assert len(motor_list) > 0
        assert len(motor_list) == len(motor_offset) == len(motor_direction), (
            f"len{len(motor_list)}, len{len(motor_offset)}, len{len(motor_direction)}"
        )
        self.motor_list = motor_list
        self.motor_offset = np.array(motor_offset)
        self.motor_direction = np.array(motor_direction)
        self.channel = channel
        self.motor_chain_name = motor_chain_name
        logging.info(f"Channel: {channel}, Bitrate: {bitrate}")
        if "can" in channel:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bustype="socketcan",
                receive_mode=receive_mode,
                name=motor_chain_name,
                control_mode=control_mode,
                use_buffered_reader=use_buffered_reader,
            )
        else:
            self.motor_interface = DMSingleMotorCanInterface(
                channel=channel,
                bitrate=bitrate,
                name=motor_chain_name,
                use_buffered_reader=use_buffered_reader,
            )
        self.state = None
        self.state_lock = threading.Lock()

        self.same_bus_device_states = None

        self.same_bus_device_lock = threading.Lock()
        if get_same_bus_device_driver is not None:
            self.same_bus_device_driver = get_same_bus_device_driver(self.motor_interface)
        else:
            self.same_bus_device_driver = None

        self.absolute_positions = None
        self.control_thread = None
        self._motor_on()
        starting_command = []
        for motor_state in self.state:
            starting_command.append(MotorCmd(torque=motor_state.torque))
        logging.info(f"Initializing motorchain with starting command: {starting_command}")
        self.commands = starting_command
        self.command_lock = threading.Lock()

        self.start_thread_flag = start_thread
        if start_thread:
            self.start_thread()

    def __repr__(self) -> str:
        return f"DMChainCanInterface(channel={self.channel})"

    def _update_absolute_positions(self, motor_feedback: List[MotorInfo]) -> None:
        init_mode = False
        if self.absolute_positions is None:
            self.absolute_positions = np.zeros(len(self.motor_list))
            init_mode = True

        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            const = MotorType.get_motor_constants(motor_type)
            position_min = const.POSITION_MIN
            position_max = const.POSITION_MAX
            position_range = position_max - position_min

            # Current position from feedback
            current_position = motor_feedback[idx].position

            # Previous absolute position
            previous_position = self.absolute_positions[idx]

            # Handle wrap-around
            delta_position = current_position - (previous_position % position_range)
            if delta_position > position_range / 2:  # Wrap backward
                delta_position -= position_range
            elif delta_position < -position_range / 2:  # Wrap forward
                delta_position += position_range

            if init_mode:
                self.absolute_positions[idx] = current_position
            else:
                self.absolute_positions[idx] += delta_position

    def __len__(self):
        return len(self.motor_list)

    def _joint_position_real_to_sim(self, joint_position_real: float) -> float:
        return (joint_position_real - self.motor_offset) * self.motor_direction

    def _joint_position_real_to_sim_idx(self, joint_position_real: float, idx: int) -> float:
        return (joint_position_real - self.motor_offset[idx]) * self.motor_direction[idx]

    def _joint_position_sim_to_real_idx(self, joint_position_sim: float, idx: int) -> float:
        return joint_position_sim * self.motor_direction[idx] + self.motor_offset[idx]

    def _motor_on(self) -> None:
        motor_feedback = []
        for _ in range(7):
            self.motor_interface.try_receive_message(timeout=0.001)
        for motor_id, motor_type in self.motor_list:
            logging.info(f"Turning on motor_id: {motor_id}, motor_type: {motor_type}")
            time.sleep(0.003)
            motor_feedback.append(self.motor_interface.motor_on(motor_id, motor_type))
        self._update_absolute_positions(motor_feedback)
        self.state = motor_feedback
        self.running = True
        logging.info("starting separate thread for control loop")

    def start_thread(self) -> None:
        # clean error again for motor with timeout enabled
        self._motor_on()
        self.control_thread = threading.Thread(target=self._set_torques_and_update_state)
        self.control_thread.start()
        time.sleep(0.1)
        while self.state is None:
            time.sleep(0.1)
            logging.info("waiting for the first state")

    def _set_torques_and_update_state(self) -> None:
        """
        Control loop for updating motor torques and states at a fixed frequency.
        If step_time > EXPECTED_CONTROL_PERIODs, it will report the number of step_time > EXPECTED_CONTROL_PERIODs and mean step_time every REPORT_INTERVAL seconds.
        """
        last_step_time = time.time()
        step_time_exceed_count = 0
        step_time_sum = 0.0
        step_time_count = 0
        report_start_time = time.time()
        with RateRecorder(name=self) as rate_recorder:
            while self.running:
                try:
                    # Maintain control frequency
                    while time.time() - last_step_time < CONTROL_PERIOD - 0.001:
                        time.sleep(0.001)
                    curr_time = time.time()
                    step_time = curr_time - last_step_time
                    last_step_time = curr_time

                    # Statistics
                    step_time_sum += step_time
                    step_time_count += 1
                    if step_time > EXPECTED_CONTROL_PERIOD:
                        step_time_exceed_count += 1

                    # If step_time > EXPECTED_CONTROL_PERIOD, report every REPORT_INTERVAL seconds
                    if step_time_exceed_count > 0 and curr_time - report_start_time >= REPORT_INTERVAL:
                        mean_step_time = step_time_sum / step_time_count if step_time_count > 0 else 0.0
                        logging.info(
                            f"[{self} {REPORT_INTERVAL}s Report] step_time > {EXPECTED_CONTROL_PERIOD}s: {step_time_exceed_count} times, mean step_time: {mean_step_time:.6f} s"
                        )
                        step_time_exceed_count = 0
                        step_time_sum = 0.0
                        step_time_count = 0
                        report_start_time = curr_time

                    # Update state
                    with self.command_lock:
                        motor_feedback = self._set_commands(self.commands)
                        errors = np.array([motor_feedback[i].error_code != "0x1" for i in range(len(motor_feedback))])
                        if np.any(errors):
                            self.running = False
                            logging.error(f"motor errors: {errors}")
                            raise Exception("motors have errors, stopping control loop")

                    with self.state_lock:
                        self.state = motor_feedback
                        self._update_absolute_positions(motor_feedback)
                    if self.same_bus_device_driver is not None:
                        time.sleep(0.001)
                        with self.same_bus_device_lock:
                            # assume the same bus device is a passive input device (no commands to send) for now.
                            self.same_bus_device_states = self.same_bus_device_driver.read_states()
                    time.sleep(0.0005)  # this is necessary, else the locks will not be released
                    rate_recorder.track()
                except Exception as e:
                    print(f"DM Error in control loop: {e}")
                    self.running = False
                    raise e

    def _set_commands(self, commands: List[MotorCmd]) -> List[MotorInfo]:
        motor_feedback = []
        for idx, motor_info in enumerate(self.motor_list):
            motor_id, motor_type = motor_info
            torque = commands[idx].torque * self.motor_direction[idx]
            pos = self._joint_position_sim_to_real_idx(commands[idx].pos, idx)

            vel = commands[idx].vel * self.motor_direction[idx]
            kp = commands[idx].kp
            kd = commands[idx].kd
            try:
                fd_back = self.motor_interface.set_control(
                    motor_id=motor_id,
                    motor_type=motor_type,
                    pos=pos,
                    vel=vel,
                    kp=kp,
                    kd=kd,
                    torque=torque,
                )
            except Exception as e:
                logging.error(f"{idx}th motor at DMChainCanInterface {self} failed with info {motor_info}")
                raise e

            motor_feedback.append(fd_back)
        return motor_feedback

    def read_states(self, torques: Optional[np.ndarray] = None) -> List[MotorInfo]:
        motor_infos = []
        with self.state_lock:
            for idx in range(len(self.motor_list)):
                state = self.state[idx]
                motor_infos.append(
                    MotorInfo(
                        id=state.id,
                        error_code=state.error_code,
                        target_torque=torques[idx] if torques is not None else 0.0,
                        vel=state.velocity * self.motor_direction[idx],
                        eff=state.torque * self.motor_direction[idx],
                        pos=self._joint_position_real_to_sim_idx(self.absolute_positions[idx], idx),
                        temp_rotor=state.temperature_rotor,
                        temp_mos=state.temperature_mos,
                    )
                )
        return motor_infos

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
        get_state: bool = True,
    ) -> List[MotorInfo]:
        command = []
        for idx in range(len(self.motor_list)):
            command.append(
                MotorCmd(
                    torque=torques[idx],
                    pos=pos[idx] if pos is not None else 0.0,
                    vel=vel[idx] if vel is not None else 0.0,
                    kp=kp[idx] if kp is not None else 0.0,
                    kd=kd[idx] if kd is not None else 0.0,
                )
            )
        with self.command_lock:
            self.commands = command
        if get_state:
            return self.read_states(torques=torques)

    def get_same_bus_device_states(self) -> Any:
        with self.same_bus_device_lock:
            return self.same_bus_device_states

    def close(self) -> None:
        logging.info(f"Closing DMChainCanInterface {self.motor_chain_name}...")
        self.running = False
        # Wait for control thread to stop
        if self.control_thread is not None and self.control_thread.is_alive():
            self.control_thread.join(timeout=1.0)
            if self.control_thread.is_alive():
                logging.warning(f"Control thread for {self.motor_chain_name} did not stop in time")
        # Turn off all motors
        for motor_id, motor_type in self.motor_list:
            try:
                self.motor_interface.motor_off(motor_id)
                time.sleep(0.05)
            except Exception as e:
                logging.warning(f"Failed to turn off motor {motor_id}: {e}")
        logging.info(f"DMChainCanInterface {self.motor_chain_name} closed")


class MultiDMChainCanInterface(MotorChain):
    """Class for interfacing with multiple asynchronous CAN interfaces."""

    def __init__(
        self,
        interfaces: List[DMChainCanInterface],
    ):
        self.interfaces = interfaces

    def __len__(self):
        return sum([len(inter) for inter in self.interfaces])

    def set_commands(
        self,
        torques: np.ndarray,
        pos: Optional[np.ndarray] = None,
        vel: Optional[np.ndarray] = None,
        kp: Optional[np.ndarray] = None,
        kd: Optional[np.ndarray] = None,
    ) -> List[MotorInfo]:
        start_idx = 0
        motor_infos = []
        for inter in self.interfaces:
            inter_len = len(inter)
            end_idx = start_idx + inter_len
            inter_torques = torques[start_idx:end_idx]
            inter_pos = pos[start_idx:end_idx] if pos is not None else None
            inter_vel = vel[start_idx:end_idx] if vel is not None else None
            inter_kp = kp[start_idx:end_idx] if kp is not None else None
            inter_kd = kd[start_idx:end_idx] if kd is not None else None
            infos = inter.set_commands(inter_torques, inter_pos, inter_vel, inter_kp, inter_kd)
            motor_infos.extend(infos)
            start_idx = end_idx
        return motor_infos


if __name__ == "__main__":
    import argparse

    from i2rt.utils.utils import override_log_level

    override_log_level(level=logging.INFO)

    args = argparse.ArgumentParser()
    args.add_argument("--channel", type=str, default="can0")
    args.add_argument("--print_state", action="store_true")
    args.add_argument("--print_pos", action="store_true")

    args = args.parse_args()
    channel = args.channel
    motor_chain_name = "yam_real"
    motor_list = [
        [0x01, "DM4310"],
        [0x02, "DM4310"],
        [0x03, "DM4310"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
        [0x07, "DM4310"],
    ]
    motor_offsets = [0] * len(motor_list)
    motor_directions = [1] * len(motor_list)
    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name,
        receive_mode=ReceiveMode.p16,
    )
    while True:
        motor_chain.set_commands(np.zeros(len(motor_list)))
        if args.print_state:
            print(motor_chain.read_states())
        if args.print_pos:
            print([state.pos for state in motor_chain.read_states()])
        time.sleep(0.1)
