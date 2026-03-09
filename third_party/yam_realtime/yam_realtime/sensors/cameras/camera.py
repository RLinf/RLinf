import threading
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Protocol, Tuple

import numpy as np

from yam_realtime.utils.portal_utils import remote


@dataclass
class IMUData:
    timestamp: float  # relative timestamp in ms
    acceleration: Optional[Tuple[float, float, float]] = None  # 3D acceleration [x, y, z]
    gyroscope: Optional[Tuple[float, float, float]] = None  # 3D gyroscope [x, y, z]


@dataclass
class CameraSpec:
    name: str  # Name of the camera
    shape: Tuple[int, int, int]  # Shape of the image (height, width, channels)
    dtype: np.dtype  # Data type of the image


@dataclass
class CameraData:
    images: Dict[str, np.ndarray]  # Named dict of multiple arrays
    timestamp: float  # milliseconds unit
    calibration_data: Optional[dict] = None
    imu_data: Optional[IMUData] = None  # Optional IMU data
    other_sensors: Optional[dict] = None  # Optional dictionary for additional sensors


class CameraDriver(Protocol):
    """Camera protocol.

    A protocol for a camera driver. This is used to abstract the camera from the rest of the code.
    """

    def read(self) -> CameraData:
        """Read a frame(RGB) from the camera.

        Returns:
            CameraData: The data read from the camera.
        """
        ...

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        """Read calibration data from the camera.

        Returns:
            dict: The calibration data.
        """
        ...

    def get_camera_info(self) -> Dict[str, Any]:
        """Retrieve camera information including device ID, resolution, FPS, and exposure settings.

        Returns:
            dict: A dictionary containing the following keys:
                - device_id: The identifier for the camera.
                - width: The camera width
                - height: The camera height
                - fps: The frames per second setting.
                - auto_exposure: The auto exposure flag.
                - exposure_value: The exposure value setting.
        """
        ...

    def stop(self) -> None:
        """Stop the camera."""
        ...


@dataclass
class CameraNode:
    camera: CameraDriver
    timeout_sec: float = 0.2  # configurable timeout

    def __post_init__(self):
        self.latest_data: Optional[CameraData] = None
        self.last_update_time: Optional[float] = None
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.polling_thread = threading.Thread(target=self._poll_image, daemon=True)
        self.polling_thread.start()
        self.exception: Optional[Exception] = None

    def _poll_image(self) -> None:
        print(f"stopevent: {self.stop_event.is_set()}")
        while not self.stop_event.is_set():
            try:
                # with Timeout(1.5):
                latest_data = self.camera.read()
                with self.lock:
                    self.latest_data = latest_data
                    self.last_update_time = time.time()
                time.sleep(0.004)
            except Exception as e:
                self.exception = e
                print(f"Error polling server: {e}")

    @remote()
    def read(self) -> Dict[str, Any]:
        if self.exception is not None:
            raise self.exception
        # block until the first image is received``
        while self.latest_data is None:
            if not self.polling_thread.is_alive():
                raise RuntimeError("Polling thread died before first image was received")
            print(f"waiting for data in camera {self.camera}")
            time.sleep(0.1)

        with self.lock:
            if self.last_update_time is None:
                raise RuntimeError("No data received yet")

            if time.time() - self.last_update_time > self.timeout_sec:
                raise TimeoutError(f"No new camera data for {self.camera} in the last {self.timeout_sec} seconds")

            return self._get_latest_data()

    def _get_latest_data(self) -> Dict[str, Any]:
        assert self.latest_data is not None, "latest_data should not be None at this point"
        return dict(
            images=self.latest_data.images,
            timestamp=self.latest_data.timestamp,
        )

    @remote(serialization_needed=True)
    def get_camera_info(self) -> Dict[str, Any]:
        return self.camera.get_camera_info()

    @remote()
    def close(self) -> None:
        self.stop_event.set()
        self.camera.stop()
        self.polling_thread.join()


@dataclass
class DummyCamera(CameraDriver):
    """A dummy camera for testing."""

    name: Optional[str] = None
    camera_specs: Optional[List[CameraSpec]] = None

    def __post_init__(self):
        if self.camera_specs is None:
            self.camera_specs = [CameraSpec(name=f"dummy_{i}", shape=(480, 640, 3), dtype=np.uint8) for i in range(2)]  # type: ignore

    def __repr__(self) -> str:
        return f"DummyCamera({self.name})"

    def read(self) -> CameraData:
        """Read a frame from the camera.

        Returns:
            CameraData: The data read from the camera.
        """
        images = {}
        assert self.camera_specs is not None, "camera_specs should not be None"
        for camera_sepc in self.camera_specs:
            image = np.random.randint(0, 255, camera_sepc.shape, dtype=camera_sepc.dtype)
            images[camera_sepc.name] = image
        return CameraData(images=images, timestamp=time.time())

    def read_calibration_data_intrinsics(self) -> Dict[str, Any]:
        """Read calibration data from the camera.

        Returns:
            dict: The calibration data.
        """
        result = {}
        assert self.camera_specs is not None, "camera_specs should not be None"
        for camera_spec in self.camera_specs:
            result[camera_spec.name] = {"K": np.eye(3), "D": np.random.rand(5)}
        return result

    def get_camera_info(self) -> Dict[str, Any]:
        """Get camera information.

        Returns:
            dict: Camera information.
        """
        return {
            "device_id": "dummy",
            "width": 640,
            "height": 480,
            "fps": 30,
            "auto_exposure": True,
            "exposure_value": 100,
        }

    def stop(self) -> None:
        print("stop dummy camera")


if __name__ == "__main__":
    cfg = """
    top_camera:
        _target_: yam_realtime.camera.camera.CameraNode
        camera:
            _target_: yam_realtime.camera.opencv_camera.OpencvCamera
            device_path: "/dev/video0"
            camera_type: "realsense_camera"
    """
    import yaml
    from tqdm import tqdm

    from yam_realtime.utils.portal_utils import launch_remote_get_local_handler

    camera_config = yaml.safe_load(cfg)
    ps, clients = [], []

    for cam_name, cam_cfg in camera_config.items():
        print(f"launching camera {cam_name} with config: {cam_cfg}")
        p, client = launch_remote_get_local_handler(cam_cfg)
        ps.append(p)
        clients.append(client)

    for _ in tqdm(range(100)):
        dats = [c.read() for c in clients]
        time.sleep(0.1)

    for c in clients:
        c.close()
    for p in ps:
        p.kill()
