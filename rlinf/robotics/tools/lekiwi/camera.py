# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Open the LeKiwi camera and report the frames it actually delivers.

The camera serial goes into the robot config, so this command also prints the
identifiers :meth:`WebcamCamera.discover` finds on this machine. A device can
open and still deliver nothing, which is why it captures real frames instead of
only opening the device.
"""

from __future__ import annotations

import argparse
import logging
import time

import numpy as np

from rlinf.robotics.parts.cameras.base import CameraInfo
from rlinf.robotics.parts.cameras.webcam import WebcamCamera

from ._support import cli_logger

#: Frame count used to estimate a frame rate.
SAMPLE_FRAMES = 10


def pick_device(requested: str | None, logger: logging.Logger) -> str:
    """Return the device to open, listing the alternatives when ambiguous."""
    found = sorted(WebcamCamera.discover())
    if not found:
        raise RuntimeError(
            "No webcam found. On Linux check /dev/v4l/by-id/ and lsusb; "
            "elsewhere close any other program using the camera."
        )
    logger.info("Webcams found: %s", ", ".join(found))
    if requested is not None:
        return requested
    if len(found) > 1:
        raise RuntimeError(
            "Several webcams found; pass --device with one of: " + ", ".join(found)
        )
    return found[0]


def measure_frames(
    camera: WebcamCamera, frames: int, logger: logging.Logger
) -> tuple[np.ndarray, float]:
    """Capture ``frames`` and return the last one with the observed frame rate."""
    started = time.monotonic()
    last = None
    for _ in range(frames):
        last = camera.get_observation()["frame"]
    elapsed = time.monotonic() - started
    logger.info(
        "Captured %d frames in %.2f s (%.1f fps)",
        frames,
        elapsed,
        frames / elapsed if elapsed > 0 else 0.0,
    )
    return last, frames / elapsed if elapsed > 0 else 0.0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the device, the requested resolution, and an optional still."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        help="Camera serial or index; defaults to the only webcam found",
    )
    parser.add_argument("--width", type=int, default=640, help="Frame width")
    parser.add_argument("--height", type=int, default=480, help="Frame height")
    parser.add_argument("--fps", type=int, default=30, help="Requested frame rate")
    parser.add_argument(
        "--save",
        metavar="PATH",
        help="Write the last frame to this image file, e.g. check.png",
    )
    args = parser.parse_args(argv)
    if args.width <= 0 or args.height <= 0:
        parser.error("--width and --height must be greater than zero")
    if args.fps <= 0:
        parser.error("--fps must be greater than zero")
    return args


def main(argv: list[str] | None = None) -> int:
    """Check the camera end to end: discover, open, capture, report."""
    args = parse_args(argv)
    logger = cli_logger()
    camera = None
    try:
        device = pick_device(args.device, logger)
        camera = WebcamCamera(
            CameraInfo(
                name="front",
                serial_number=device,
                camera_type="webcam",
                resolution=(args.width, args.height),
                fps=args.fps,
            )
        )
        camera.connect()
        if not camera.is_ready(timeout=3.0):
            raise RuntimeError(
                f"{device} opened but delivered no frame within 3 s; "
                "check the cable and that no other program holds the camera."
            )
        logger.info("Using %s at %dx%d", device, args.width, args.height)
        frame, _fps = measure_frames(camera, SAMPLE_FRAMES, logger)
        logger.info(
            "Frame shape %s, dtype %s, mean brightness %.1f",
            frame.shape,
            frame.dtype,
            float(np.mean(frame)),
        )
        if frame.shape[:2] != (args.height, args.width):
            logger.warning(
                "Expected %dx%d but the device delivered %dx%d; OpenCV resizes it, "
                "so pick a resolution the camera supports natively.",
                args.width,
                args.height,
                frame.shape[1],
                frame.shape[0],
            )
        if args.save:
            import cv2

            cv2.imwrite(args.save, frame)
            logger.info("Saved the last frame to %s", args.save)
    except KeyboardInterrupt:
        logger.info("Camera check interrupted.")
        return 1
    except Exception:  # noqa: BLE001 - report the hardware failure and exit non-zero
        logger.exception("Camera check failed")
        return 1
    finally:
        if camera is not None and camera.is_connected:
            camera.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
