import argparse

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

args = argparse.ArgumentParser()
args.add_argument("--channel", type=str, default="can0")
args.add_argument("--motor_id", type=int, default=1)

args = args.parse_args()
motor_control_interface = DMSingleMotorCanInterface(
    channel=args.channel, bustype="socketcan", control_mode=ControlMode.MIT
)

motor_id = args.motor_id
motor_control_interface.motor_on(motor_id, MotorType.DM4310)
current_position = motor_control_interface.set_control(motor_id, MotorType.DM4310, 0, 0, 0, 0, 0).position
print(f"current position: {current_position}")

motor_control_interface.save_zero_position(motor_id)
print("set zero .....")

current_position = motor_control_interface.set_control(motor_id, MotorType.DM4310, 0, 0, 0, 0, 0).position
print(f"after set to zero position, current position: {current_position}")
