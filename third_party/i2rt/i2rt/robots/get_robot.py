import logging
import time
from functools import partial

import numpy as np

from i2rt.motor_drivers.dm_driver import (
    CanInterface,
    DMChainCanInterface,
    EncoderChain,
    PassiveEncoderReader,
    ReceiveMode,
)
from i2rt.robots.motor_chain_robot import MotorChainRobot
from i2rt.robots.utils import GripperType


def get_encoder_chain(can_interface: CanInterface) -> EncoderChain:
    passive_encoder_reader = PassiveEncoderReader(can_interface)
    return EncoderChain([0x50E], passive_encoder_reader)


def get_yam_robot(
    channel: str = "can0",
    gripper_type: GripperType = GripperType.CRANK_4310,
    zero_gravity_mode: bool = True,
    gripper_limits: list = None,
    gripper_kp: float = None,
    gripper_kd: float = None,
    limit_gripper_force: float = None,
) -> MotorChainRobot:
    with_gripper = True
    with_teaching_handle = False
    if gripper_type == GripperType.YAM_TEACHING_HANDLE:
        with_gripper = False
        with_teaching_handle = True
    if gripper_type == GripperType.NO_GRIPPER:
        with_gripper = False
        with_teaching_handle = False

    model_path = gripper_type.get_xml_path()
    motor_list = [
        [0x01, "DM4340"],
        [0x02, "DM4340"],
        [0x03, "DM4340"],
        [0x04, "DM4310"],
        [0x05, "DM4310"],
        [0x06, "DM4310"],
    ]
    motor_offsets = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    joint_limits = np.array([[-2.617, 3.13], [0, 3.65], [0.0, 3.13], [-1.57, 1.57], [-1.57, 1.57], [-2.09, 2.09]])
    joint_limits[:,0] += -0.15 # add some buffer to the joint limits
    joint_limits[:,1] += 0.15

    motor_directions = [1, 1, 1, 1, 1, 1]
    kp = np.array([80, 80, 80, 40, 10, 10])
    kd = np.array([5, 5, 5, 1.5, 1.5, 1.5])
    # kp = np.array([5, 5, 5, 3, 2, 2]) 
    # kd = np.array([0.5, 0.5, 0.5, 0.6, 0.4, 0.4])
    
    if with_gripper:
        motor_type = gripper_type.get_motor_type()
        default_gripper_kp, default_gripper_kd = gripper_type.get_motor_kp_kd()
        final_gripper_kp = gripper_kp if gripper_kp is not None else default_gripper_kp
        final_gripper_kd = gripper_kd if gripper_kd is not None else default_gripper_kd
        assert motor_type != ""
        logging.info(
            f"adding gripper motor with type: {motor_type}, gripper_kp: {final_gripper_kp}, gripper_kd: {final_gripper_kd}"
        )
        motor_list.append([0x07, motor_type])
        motor_offsets.append(0.0)
        motor_directions.append(1)
        kp = np.concatenate([kp, np.array([final_gripper_kp])])
        kd = np.concatenate([kd, np.array([final_gripper_kd])])

    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        start_thread=False,
    )
    motor_states = motor_chain.read_states()
    print(f"motor_states: {motor_states}")
    motor_chain.close()

    current_pos = [m.pos for m in motor_states]
    logging.info(f"current_pos: {current_pos}")

    gripper_motor_idx = 6 if with_gripper else None

    for idx, motor_state in enumerate(motor_states):
        # Skip offset correction for gripper motor — its position can naturally exceed ±π
        if idx == gripper_motor_idx:
            continue
        motor_position = motor_state.pos
        # if not within -pi to pi, set to the nearest equivalent position
        if motor_position < -np.pi:
            logging.info(f"motor {idx} is at {motor_position}, adding {2 * np.pi}")
            extra_offset = -2 * np.pi
        elif motor_position > np.pi:
            logging.info(f"motor {idx} is at {motor_position}, subtracting {2 * np.pi}")
            extra_offset = +2 * np.pi
        else:
            extra_offset = 0.0
        motor_offsets[idx] += extra_offset

    time.sleep(0.5)
    logging.info(f"adjusted motor_offsets: {motor_offsets}")

    motor_chain = DMChainCanInterface(
        motor_list,
        motor_offsets,
        motor_directions,
        channel,
        motor_chain_name="yam_real",
        receive_mode=ReceiveMode.p16,
        get_same_bus_device_driver=get_encoder_chain if with_teaching_handle else None,
        use_buffered_reader=False,
    )
    motor_states = motor_chain.read_states()
    logging.info(f"YAM initial motor_states: {motor_states}")
    get_robot = partial(
        MotorChainRobot,
        motor_chain=motor_chain,
        xml_path=model_path,
        use_gravity_comp=True,
        gravity_comp_factor=1.3,
        joint_limits=joint_limits,
        kp=kp,
        kd=kd,
        zero_gravity_mode=zero_gravity_mode,
    )

    if with_gripper:
        # Use YAML overrides if provided, otherwise fall back to GripperType defaults
        final_gripper_limits = gripper_limits if gripper_limits is not None else gripper_type.get_gripper_limits()
        final_limit_gripper_force = limit_gripper_force if limit_gripper_force is not None else 50.0
        return get_robot(
            gripper_index=6,
            gripper_limits=final_gripper_limits,
            enable_gripper_calibration=gripper_type.get_gripper_needs_calibration(),
            gripper_type=gripper_type,
            limit_gripper_force=final_limit_gripper_force,
        )
    else:
        return get_robot()
