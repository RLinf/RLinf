import asyncio
import base64
import tempfile
from typing import TYPE_CHECKING, Any, Optional

from omegaconf import DictConfig

from rlinf.envs.adb.adb_env import ADBEnv
from rlinf.scheduler import Channel, Worker

if TYPE_CHECKING:
    from rlinf.scheduler.hardware import ADBHWInfo


def encode_image(image_path: str) -> str:
    """
    Encode image to base64 string.

    Args:
        image_path: Path to the image file

    Returns:
        Base64 encoded string
    """
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


class PhoneWorker(Worker):
    """Worker that manages one or more ADB-connected phone/emulator devices.

    This worker can manage multiple ADB devices simultaneously, routing requests
    to the correct device via device_key (local device index within this worker).

    Key concepts:
        - device_key: Local index (0, 1, 2, ...) of the device within this worker.
                     Used as channel key for routing messages.
        - global_device_key: Combination of worker rank and local device index,
                            e.g., "0_0", "0_1", "1_0" for unique identification.
    """

    def __init__(self, cfg: DictConfig):
        Worker.__init__(self)

        self.cfg = cfg
        if not self.hardware_infos:
            raise ValueError(
                "PhoneWorker requires hardware_infos to bind ADB device(s)."
            )

        # Create ADBEnv for each hardware info
        # Key is local device index (0, 1, 2, ...)
        self.phone_envs: dict[int, ADBEnv] = {}

        for idx, hw_info in enumerate(self.hardware_infos):
            hw_info: "ADBHWInfo"
            self.phone_envs[idx] = ADBEnv(
                adb_path=hw_info.config.adb_path,
                device_id=hw_info.config.device_id,
            )

        self.log_info(f"PhoneWorker[{self._rank}] initialized with {len(self.phone_envs)} devices")

        self._num_devices = len(self.phone_envs)

    def _get_env(self, device_key: int) -> ADBEnv:
        """Get ADBEnv by local device key.

        Args:
            device_key: device id".
        """
        return self.phone_envs[device_key]

    async def get_num_devices(self) -> int:
        """Get the number of devices managed by this worker."""
        return self._num_devices

    def init_worker(self):
        pass

    def make_device_key(self, worker_rank: int, device_idx: int) -> str:
        """Create a globally unique device key.

        Args:
            worker_rank: The rank of the PhoneWorker.
            device_idx: The local device index within that worker.

        Returns:
            Device key in format "{worker_rank}_{device_idx}".
        """
        return f"{worker_rank}_{device_idx}"

    async def _handle_device_loop(
        self,
        device_idx: int,
        input_channel: Channel,
        output_channel: Channel,
    ):
        """Handle the interaction loop for a single device.

        Args:
            device_idx: Local device index.
            input_channel: Channel to receive requests.
            output_channel: Channel to send responses.
        """
        # Use "{worker_rank}_{device_idx}" as channel key to be globally unique
        # since all workers share the same channel
        channel_key = self.make_device_key(self._rank, device_idx)
        self.log_info(
            f"PhoneWorker[{self._rank}] device {device_idx} "
            f"listening on key '{channel_key}'"
        )

        while True:
            input_message = await input_channel.get(
                key=channel_key, async_op=True
            ).async_wait()

            if input_message is None:
                self.log_info(
                    f"PhoneWorker[{self._rank}] device {device_idx} "
                    f"received termination signal"
                )
                break

            self.log_info(f"phone[{channel_key}] get message: {input_message}")

            # Process request (blocking ADB operations run in executor)
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._process_device_request,
                device_idx,
                input_message,
            )

            await output_channel.put(
                response, key=channel_key, async_op=True
            ).async_wait()

    async def interact_multi_device(
        self, input_channel: Channel, output_channel: Channel
    ):
        """Concurrent interaction loop for all devices managed by this worker.

        Each device listens on its own channel key: "{worker_rank}_{device_idx}".
        This ensures globally unique keys since all workers share the same channel.

        This is the recommended method when one PhoneWorker manages multiple devices.
        """
        self.log_info(
            f"PhoneWorker[{self._rank}] starting multi-device loop "
            f"for {self._num_devices} devices"
        )

        # Create a task for each device
        tasks = [
            asyncio.create_task(
                self._handle_device_loop(device_key, input_channel, output_channel)
            )
            for device_key in range(self._num_devices)
        ]

        # Wait for all device loops to complete
        await asyncio.gather(*tasks)

        self.log_info(f"PhoneWorker[{self._rank}] multi-device loop terminated")

    def interact(self, input_channel: Channel, output_channel: Channel):
        """Single device interact loop (backward compatible, uses first device)."""
        while True:
            input_message = input_channel.get()
            if input_message is None:
                break

            self.log_info(f"phone get message: {input_message}")
            if input_message["name"] == "mobile_use":
                self.execute_tool_call(input_message, device_key=0)
            else:
                assert input_message["name"] == "screenshot", (
                    f"Unknown tool: {input_message['name']}"
                )
            with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
                screenshot = self.phone_envs[0].screenshot(tmp_file.name)
                if not screenshot:
                    raise ValueError("Failed to take screenshot")
                base64_image = encode_image(tmp_file.name)
            output_channel.put(
                {
                    "type": "image",
                    "image": f"data:image/jpeg;base64,{base64_image}",
                }
            )

    def execute_tool_call(self, tool_call: dict[str, Any], device_key: int) -> bool:
        """Execute a tool call on a specific device.

        Args:
            tool_call: Tool call dict with name and arguments.
            device_key: Local device index. If None, uses the first device.
        """
        if tool_call.get("name") != "mobile_use":
            raise ValueError(f"Unknown tool: {tool_call.get('name')}")

        phone_env = self._get_env(device_key)
        arguments = tool_call.get("arguments", {})
        action = arguments.get("action", "")

        kwargs = {k: v for k, v in arguments.items() if k != "action"}
        return phone_env.execute_mobile_action(action, **kwargs)

    def _process_device_request(
        self,
        device_key: int,
        input_message: dict[str, Any],
    ) -> dict[str, Any]:
        """Process a single request for a specific device.

        Args:
            device_key: Local device index.
            input_message: The request message.

        Returns:
            Screenshot response dict.
        """
        phone_env = self._get_env(device_key)

        if input_message["name"] == "mobile_use":
            self.execute_tool_call(input_message, device_key=device_key)
        else:
            assert input_message["name"] == "screenshot", (
                f"Unknown tool: {input_message['name']}"
            )

        with tempfile.NamedTemporaryFile(suffix=".png") as tmp_file:
            screenshot = phone_env.screenshot(tmp_file.name)
            if not screenshot:
                raise ValueError(f"Failed to take screenshot from device {device_key}")
            base64_image = encode_image(tmp_file.name)

        return {
            "type": "image",
            "image": f"data:image/jpeg;base64,{base64_image}",
        }
