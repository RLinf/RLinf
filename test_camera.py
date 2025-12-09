import pyrealsense2 as rs
import time


def main():
    for device in rs.context().devices:
        serial_number = device.get_info(rs.camera_info.serial_number)
        print(serial_number)
        device.get_info(rs.camera_info.serial_number)
    # print(serial_number)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(serial_number)
    config.enable_stream(
        rs.stream.color,
            640,
            480, 
            rs.format.bgr8,
            15,
    )
    pipeline.start(config)
    
    for step in range(20):
        print(step)
        time.sleep(0.1)
        frames = pipeline.wait_for_frames()

if __name__ == "__main__":
    main()