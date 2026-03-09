import argparse

from i2rt.motor_drivers.dm_driver import ControlMode, DMSingleMotorCanInterface, MotorType

args = argparse.ArgumentParser()
args.add_argument("--channel", type=str, default="can0")
args.add_argument("--motor_id", type=int, default=-1)

args = args.parse_args()

motor_control_interface = DMSingleMotorCanInterface(
    channel=args.channel, bustype="socketcan", control_mode=ControlMode.MIT
)

if args.motor_id < 0:
    motor_ids = [1, 2, 3, 4, 5, 6, 7]
else:
    motor_ids = [args.motor_id]


online_motors = []
for motor_id in motor_ids:
    try:
        motor_info = motor_control_interface.motor_on(motor_id, MotorType.DM4310)
        print(f"motor {motor_id} info: {motor_info}")
        online_motors.append(motor_id)
        motor_control_interface.motor_off(motor_id)
    except Exception as e:
        print(f"motor {motor_id} error: {e}")


motor_control_interface.close()

print(f"online motors: {online_motors}")
