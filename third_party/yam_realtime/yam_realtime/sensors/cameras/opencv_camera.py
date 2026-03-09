import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from yam_realtime.sensors.cameras.camera import CameraData, CameraDriver


@dataclass
class OpencvCamera(CameraDriver):
    """
    bash v4l2-ctl --list-devices -> get the device id
    bash v4l2-ctl --device=/dev/video12 --list-formats-ext -> get the resolution and fps

    """

    device_path: str = "/dev/video-zed2i"
    camera_type: str = "realsense_camera"
    image_transfer_time_offset: int = 80  # ms typical transfer time, can change based on computer type and load
    resolution: Tuple[int, int] = (640, 480)
    fps: int = 30
    name: Optional[str] = None

    def __repr__(self) -> str:
        return f"OpencvCamera(device_path={self.device_path!r}, name={self.name!r}, resolution={self.resolution}, fps={self.fps})"

    def __post_init__(self):
        available_cameras = self.list_cameras()
        logging.info(f"available_cameras: {available_cameras}")

        self.cap = cv2.VideoCapture(self.device_path)
        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.resolution[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.resolution[1])

        # Try setting FPS to 30
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def list_cameras(self) -> List[int]:
        available_cameras = []
        for i in range(20):  # Check the first 20 device indices
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                available_cameras.append(i)
                cap.release()

        return available_cameras

    def read(self) -> CameraData:
        try:
            ret, frame = self.cap.read()
            capture_time_ms = time.time() * 1000
            while not ret:
                # If read failed, retry and update capture time
                ret, frame = self.cap.read()
                capture_time_ms = time.time() * 1000
                time.sleep(0.01)
        except Exception as e:
            logging.error(f"Error reading frame: {e}")
            raise e
        frame = np.ascontiguousarray(frame)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # Adjust timestamp using the offset
        timestamp_ms = capture_time_ms - self.image_transfer_time_offset
        return CameraData(images={"rgb": frame}, timestamp=timestamp_ms)

    def get_camera_info(self) -> dict:
        info = {}
        # opencv resolution does not match zed resolution, but our config is set as zed resolution ,so it can be used temporarily
        info.update(
            {
                "camera_type": self.camera_type,
                "device_path": self.device_path,
                "width": self.resolution[0],
                "height": self.resolution[1],
                "fps": self.fps,
            }
        )
        return info

    def stop(self) -> None:
        self.cap.release()

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        raise NotImplementedError(f"Calibration data reading is not implemented for {self}")


if __name__ == "__main__":
    import argparse

    from yam_realtime.camera.camera_utils import plot_camera_read

    parser = argparse.ArgumentParser()
    parser.add_argument("--device_path", type=str, default="/dev/video-zed2i")
    parser.add_argument("--show_video", action="store_true")
    args = parser.parse_args()

    camera = OpencvCamera(device_path=args.device_path)

    if args.show_video:
        plot_camera_read(camera)
    else:
        while True:
            data = camera.read()
            print(data)
            time.sleep(1 / camera.fps)
