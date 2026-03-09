import argparse
import portal
from i2rt.flow_base.flow_base_controller import BASE_DEFAULT_PORT
import numpy as np
import time
import threading
import sys

class FlowBaseClient:
    def __init__(self, host: str = "localhost"):
        self.client = portal.Client(f"{host}:{BASE_DEFAULT_PORT}")
        self.command = {'target_velocity': np.array([0.0, 0.0, 0.0]), 'frame': 'local'}
        self._lock = threading.Lock()
        self.running = True
        self._thread = threading.Thread(target=self._update_velocity)
        self._thread.start()

    def _update_velocity(self):
        while self.running:
            with self._lock:
                self.client.set_target_velocity(self.command).result()
            time.sleep(0.02)

    def get_odometry(self):
        return self.client.get_odometry({}).result()


    def reset_odometry(self):
        return self.client.reset_odometry({}).result()

    def set_target_velocity(self, target_velocity: np.ndarray, frame: str = "local"):
        with self._lock:
            self.command['target_velocity'] = target_velocity
            self.command['frame'] = frame
    def close(self):
        self.running = False
        self._thread.join()
        self.client.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="localhost")
    parser.add_argument("--command", type=str, default="get_odometry")
    args = parser.parse_args()

    client = FlowBaseClient(args.host)

    if args.command == "get_odometry":
        print(client.get_odometry())
        client.close()
        exit()
    elif args.command == "reset_odometry":
        client.reset_odometry()
        client.close()
        exit()
    elif args.command == "test_command":
        client.set_target_velocity(np.array([0.0, 0.0, 0.1]), "local")
        while True:
            odo_reading = client.get_odometry()
            sys.stdout.write(
                f"\r translation: {odo_reading['translation']} rotation: {odo_reading['rotation']}"
            )
            sys.stdout.flush()
            time.sleep(0.02)
