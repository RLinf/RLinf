import struct
import time
from typing import Any, List, Optional

import can


class RawCanInterface:
    def __init__(
        self,
        channel: str = "PCAN_USBBUS1",
        bustype: str = "pcan",
        bitrate: int = 1000000,
        name: str = "default_can_interface",
    ):
        self.bus = can.interface.Bus(bustype=bustype, channel=channel, bitrate=bitrate)
        self.busstate = self.bus.state
        self.name = name

    def close(self) -> None:
        """Shut down the CAN bus."""
        self.bus.shutdown()

    def _send_message_get_response(self, id: int, data: List[int], max_retry: int = 20) -> can.Message:
        self.try_receive_message(id)
        message = can.Message(arbitration_id=id, data=data, is_extended_id=False)
        for _ in range(max_retry):
            try:
                self.bus.send(message)
                response = self._receive_message(data[0])
                return response
            except (can.CanError, AssertionError) as e:
                print(e)
                # print warning in red
                print(
                    "\033[91m"
                    + f"CAN Error {self.name}: Failed to communicate with motor {data[0]} over can bus. Retrying..."
                    + "\033[0m"
                )
            time.sleep(0.002)
        raise AssertionError(
            f"fail to communicate with the motor {id} on {self.name} at can channel {self.bus.channel_info}"
        )

    def try_receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> can.Message:
        """Try to receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message, or None if no message is received.
        """
        try:
            return self._receive_message(motor_id, timeout)
        except AssertionError:
            return None

    def _receive_message(self, motor_id: Optional[int] = None, timeout: float = 0.009) -> can.Message:
        """Receive a message from the CAN bus.

        Args:
            timeout (float): The time to wait for a message (in seconds).

        Returns:
            can.Message: The received message.

        Raises:
            AssertionError: If no message is received within the timeout.
        """
        start_time = time.time()
        while (time.time() - start_time) < timeout:
            message = self.bus.recv(timeout=0.002)
            if message:
                return message
        raise AssertionError(
            f"Failed to receive message, {self.name} motor id {motor_id} motor timeout. Check if the motor is powered on or if the motor ID exists."
        )


def bytes_to_uint32(data: bytearray) -> int:
    """Convert the last 4 bytes (data[4:8]) in a bytearray to uint32 (little-endian)."""
    return struct.unpack("<I", data[4:8])[0]


def bytes_to_float32(data: bytearray) -> float:
    """Convert the last 4 bytes (data[4:8]) in a bytearray to float32 (little-endian)."""
    return struct.unpack("<f", data[4:8])[0]


def uint32_to_bytes(value: int) -> bytearray:
    """Convert a uint32 value to a 4-byte bytearray (little-endian).

    Args:
        value (int): A uint32 value to be converted to bytes.

    Returns:
        bytearray: The corresponding 4-byte representation in little-endian order.
    """
    return bytearray(struct.pack("<I", value))


def float32_to_bytes(value: float) -> bytearray:
    """Convert a float32 value to a 4-byte bytearray (little-endian).

    Args:
        value (float): A float32 value to be converted to bytes.

    Returns:
        bytearray: The corresponding 4-byte representation in little-endian order.
    """
    return bytearray(struct.pack("<f", value))


register_addr_map = {
    "KT_value": (1, bytes_to_float32),
    "OT_value": (2, bytes_to_float32),
    "master_id": (7, bytes_to_uint32),
    "id": (8, bytes_to_uint32),
    "timeout": (9, bytes_to_uint32),
    "inertia": (12, bytes_to_float32),
    "sw_ver": (14, bytes_to_uint32),
    "flux": (19, bytes_to_float32),
    "gear_ratio": (20, bytes_to_float32),
    "gear_eff": (30, bytes_to_float32),
}

register_info_map = {"control_mode": {1: "MIT", 2: "pos_speed", 3: "speed", 4: "torque_pos"}}


def get_special_message_response(can_interface: RawCanInterface, motor_id: int, reg_name: str) -> Any:
    """Get the current value of a register from a motor.

    Args:
        can_interface (RawCanInterface): The CAN interface to use.
        motor_id (int): The ID of the motor to read the register value from.
        reg_name (str): The name of the register to read.

    Returns:
        Any: The value read from the register.
    """
    assert reg_name in register_addr_map, f"reg_name {reg_name} not in register_addr_map"
    reg_id, convert_func = register_addr_map[reg_name]
    for _ in range(3):
        try:
            message = can_interface._send_message_get_response(
                0x7FF, [motor_id, 0x00, 0x33, reg_id, 0x00, 0x00, 0x00, 0x00], max_retry=20
            )
            return convert_func(message.data)
        except Exception as e:
            can_interface.try_receive_message(motor_id)
    raise Exception(f"Failed to read {reg_name} of motor {motor_id} after 3 retries")


def write_special_message(can_interface: RawCanInterface, motor_id: int, reg_name: str, data: Any) -> Any:
    """Write a value to a register of a motor.

    Args:
        can_interface (RawCanInterface): The CAN interface to use.
        motor_id (int): The ID of the motor to write the register value for.
        reg_name (str): The name of the register to write.
        data (Any): The value to write to the register.

    Returns:
        Any: The value read from the register after writing.
    """
    assert reg_name in register_addr_map, f"reg_name {reg_name} not in register_addr_map"
    reg_id, convert_func = register_addr_map[reg_name]
    if convert_func is bytes_to_uint32:
        byte_list = uint32_to_bytes(data)
    elif convert_func is bytes_to_float32:
        byte_list = float32_to_bytes(data)
    for _ in range(3):
        try:
            message = can_interface._send_message_get_response(
                0x7FF, [motor_id, 0x00, 0x55, reg_id] + list(byte_list), max_retry=20
            )
            result = convert_func(message.data)
            return result
        except Exception as e:
            can_interface.try_receive_message(motor_id)


def save_to_memory(can_interface: RawCanInterface, motor_id: int, reg_name: str) -> can.Message:
    """Save the current value of a register to the motor's memory.

    Args:
        can_interface (RawCanInterface): The CAN interface to use.
        motor_id (int): The ID of the motor to save the register value for.
        reg_name (str): The name of the register to save.

    Returns:
        can.Message: The response message from the motor.
    """
    assert reg_name in register_addr_map, f"reg_name {reg_name} not in register_addr_map"
    reg_id, convert_func = register_addr_map[reg_name]
    message = can_interface._send_message_get_response(
        0x7FF, [motor_id, 0x00, 0xAA, reg_id, 0x00, 0x00, 0x00, 0x00], max_retry=20
    )
    return message
