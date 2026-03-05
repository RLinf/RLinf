# Copyright 2025 The RLinf Authors.
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

import asyncio
import copy
import json
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, Optional

from omegaconf import DictConfig

from rlinf.scheduler import Channel
from rlinf.utils.placement import ModelParallelComponentPlacement
from rlinf.workers.agent.agent_loop import AgentLoopOutput, AgentLoopWorker


@dataclass
class DevicePoolStats:
    """Statistics for the device pool."""

    total_devices: int = 0
    available_devices: int = 0
    in_use_devices: int = 0
    total_requests_processed: int = 0
    pending_requests: int = 0


def get_system_prompt() -> str:
    """Get the system prompt for mobile agent."""
    return """

# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{"type": "function", "function": {"name": "mobile_use", "description": "Use a touchscreen to interact with a mobile device, and take screenshots.\\n* This is an interface to a mobile device with touchscreen. You can perform actions like clicking, typing, swiping, etc.\\n* Some applications may take time to start or process actions, so you may need to wait and take successive screenshots to see the results of your actions.\\n* The screen's resolution is 999x999.\\n* Make sure to click any buttons, links, icons, etc with the cursor tip in the center of the element. Don't click boxes on their edges unless asked.", "parameters": {"properties": {"action": {"description": "The action to perform. The available actions are:\\n* `click`: Click the point on the screen with coordinate (x, y).\\n* `long_press`: Press the point on the screen with coordinate (x, y) for specified seconds.\\n* `swipe`: Swipe from the starting point with coordinate (x, y) to the end point with coordinates2 (x2, y2).\\n* `type`: Input the specified text into the activated input box.\\n* `answer`: Output the answer.\\n* `system_button`: Press the system button.\\n* `wait`: Wait specified seconds for the change to happen.\\n* `terminate`: Terminate the current task and report its completion status.", "enum": ["click", "long_press", "swipe", "type", "answer", "system_button", "wait", "terminate"], "type": "string"}, "coordinate": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=click`, `action=long_press`, and `action=swipe`.", "type": "array"}, "coordinate2": {"description": "(x, y): The x (pixels from the left edge) and y (pixels from the top edge) coordinates to move the mouse to. Required only by `action=swipe`.", "type": "array"}, "text": {"description": "Required only by `action=type` and `action=answer`.", "type": "string"}, "time": {"description": "The seconds to wait. Required only by `action=long_press` and `action=wait`.", "type": "number"}, "button": {"description": "Back means returning to the previous interface, Home means returning to the desktop, Menu means opening the application background menu, and Enter means pressing the enter. Required only by `action=system_button`", "enum": ["Back", "Home", "Menu", "Enter"], "type": "string"}, "status": {"description": "The status of the task. Required only by `action=terminate`.", "type": "string", "enum": ["success", "failure"]}}, "required": ["action"], "type": "object"}}}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{"name": <function-name>, "arguments": <args-json-object>}
</tool_call>

# Response format

Response format for every step:
1) Thought: one concise sentence explaining the next move (no multi-step reasoning).
2) Action: a short imperative describing what to do in the UI.
3) A single <tool_call>...</tool_call> block containing only the JSON: {"name": <function-name>, "arguments": <args-json-object>}.

Rules:
- Output exactly in the order: Thought, Action, <tool_call>.
- Be brief: one sentence for Thought, one for Action.
- Do not output anything else outside those three parts.
- If finishing, use action=terminate in the tool call."""


def parse_tool_call(output_text: str) -> Optional[dict[str, Any]]:
    """
    Parse tool call from LLM output.

    Args:
        output_text: Raw output from LLM

    Returns:
        Parsed tool call dictionary or None if parsing fails
    """
    try:
        # Extract tool call from XML tags
        if "<tool_call>" in output_text and "</tool_call>" in output_text:
            tool_call_str = (
                output_text.split("<tool_call>")[1].split("</tool_call>")[0].strip()
            )
            return json.loads(tool_call_str)
        else:
            print("Warning: No tool_call tags found in output")
            return None
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing tool call: {e}")
        print(f"Output text: {output_text}")
        return None


def rescale_coordinates(point: list[int], width: int, height: int) -> list[int]:
    """
    Rescale coordinates from 999x999 to actual screen resolution.

    Args:
        point: [x, y] coordinates in 999x999 space
        width: Actual screen width
        height: Actual screen height

    Returns:
        Rescaled [x, y] coordinates
    """
    return [round(point[0] / 999 * width), round(point[1] / 999 * height)]


def delete_images_in_history(history: list[dict]):
    """
    Delete images in history.
    """
    for message in history:
        assert isinstance(message, dict), f"Message is not a dict: {message}"
        message["content"] = [
            msg for msg in message["content"] if msg["type"] == "text"
        ]


class MobileAgentLoopWorker(AgentLoopWorker):
    """Agent loop worker for mobile phone tasks with VL model.

    This worker manages the multi-turn conversation loop between
    the VL model and the phone environment.

    Architecture:
        - One MobileAgentLoopWorker per node
        - Communicates with the PhoneWorker on the same node via shared channels
        - Manages a pool of devices (emulators) on the current node
        - Requests may exceed device count; uses async queue for resource management

    Device Key Format:
        "{phone_worker_rank}_{local_device_idx}" - globally unique across all nodes
        since all workers share the same channel.
    """

    def __init__(
        self,
        cfg: DictConfig,
        placement: ModelParallelComponentPlacement,
    ):
        super().__init__(cfg, placement)
        # Screen configuration
        self.screen_width = 1080
        self.screen_height = 2400
        self.max_steps = 20

        # Device pool management
        self._device_pool: Optional[asyncio.Queue[int]] = None
        self._num_devices: int = 0
        # Each agent loop worker is responsible for one phone worker.
        self._phone_worker_rank = self._rank
        self._devices_in_use: set[int] = set()
        self._pool_lock = asyncio.Lock()
        self._stats = DevicePoolStats()

    def init_mobile_worker(
        self,
        generate_input_channel: Channel,
        generate_output_channel: Channel,
        phone_input_channel: Channel,
        phone_output_channel: Channel,
        num_devices: list[int],
    ):
        """Initialize channels and device pool for mobile agent.

        Args:
            generate_input_channel: Channel to send generation requests to VL model.
            generate_output_channel: Channel to receive generation results from VL model.
            phone_input_channel: Channel to send actions to PhoneWorker (shared channel).
            phone_output_channel: Channel to receive screenshots from PhoneWorker (shared channel).
            num_devices: Number of devices (emulators) available on this node.
        """
        self.generate_input_channel = generate_input_channel
        self.generate_output_channel = generate_output_channel
        self.phone_input_channel = phone_input_channel
        self.phone_output_channel = phone_output_channel

        # Initialize device pool
        self._num_devices = num_devices[self._phone_worker_rank]
        self._device_pool = asyncio.Queue()
        for device_idx in range(self._num_devices):
            self._device_pool.put_nowait(device_idx)

        self._stats.total_devices = self._num_devices
        self._stats.available_devices = self._num_devices

        self.log_info(
            f"MobileAgentLoopWorker initialized with {self._num_devices} devices, "
            f"phone_worker_rank={self._phone_worker_rank}"
        )

    def make_device_key(self, local_device_idx: int) -> str:
        """Create a globally unique device key.

        Args:
            local_device_idx: The local device index within PhoneWorker.

        Returns:
            Device key in format "{phone_worker_rank}_{local_device_idx}".
        """
        return f"{self._phone_worker_rank}_{local_device_idx}"

    async def acquire_device(self) -> int:
        """Acquire an available device from the pool.

        This method blocks until a device becomes available.

        Returns:
            The local device index of the acquired device.
        """
        assert self._device_pool is not None, (
            "Device pool not initialized. Call init_mobile_worker() first."
        )

        async with self._pool_lock:
            self._stats.pending_requests += 1

        # This will block if no device is available
        device_idx = await self._device_pool.get()

        async with self._pool_lock:
            self._devices_in_use.add(device_idx)
            self._stats.pending_requests -= 1
            self._stats.available_devices -= 1
            self._stats.in_use_devices += 1

        return device_idx

    async def release_device(self, device_idx: int):
        """Release a device back to the pool.

        Args:
            device_idx: The local device index to release.
        """
        assert self._device_pool is not None, (
            "Device pool not initialized. Call init_mobile_worker() first."
        )

        async with self._pool_lock:
            assert device_idx in self._devices_in_use, (
                f"Trying to release device {device_idx} that is not in use"
            )
            self._devices_in_use.remove(device_idx)
            self._stats.available_devices += 1
            self._stats.in_use_devices -= 1
            self._stats.total_requests_processed += 1

        await self._device_pool.put(device_idx)

    @asynccontextmanager
    async def device_context(self):
        """Context manager for acquiring and releasing a device.

        Usage:
            async with self.device_context() as device_idx:
                # Use device_idx for rollout
                await self.run_one_query(..., device_key=device_idx)
        """
        device_idx = await self.acquire_device()
        try:
            yield device_idx
        finally:
            await self.release_device(device_idx)

    def get_pool_stats(self) -> DevicePoolStats:
        """Get current device pool statistics."""
        return DevicePoolStats(
            total_devices=self._stats.total_devices,
            available_devices=self._stats.available_devices,
            in_use_devices=self._stats.in_use_devices,
            total_requests_processed=self._stats.total_requests_processed,
            pending_requests=self._stats.pending_requests,
        )

    async def get_screenshot(self, device_key: str) -> dict:
        """Get screenshot from phone worker.

        Args:
            device_key: Device key in format "{phone_worker_rank}_{local_device_idx}".
        """
        await self.phone_input_channel.put(
            {"name": "screenshot"}, key=device_key, async_op=True
        ).async_wait()
        screenshot = await self.phone_output_channel.get(
            key=device_key, async_op=True
        ).async_wait()
        return screenshot

    async def execute_action(self, tool_call: dict, device_key: str) -> dict:
        """Execute action on phone and return screenshot.

        Args:
            tool_call: Tool call dict with name and arguments.
            device_key: Device key in format "{phone_worker_rank}_{local_device_idx}".

        Returns:
            Screenshot after action execution.
        """
        # Rescale coordinates if needed
        arguments = tool_call.get("arguments", {})
        if "coordinate" in arguments:
            arguments["coordinate"] = rescale_coordinates(
                arguments["coordinate"], self.screen_width, self.screen_height
            )
        if "coordinate2" in arguments:
            arguments["coordinate2"] = rescale_coordinates(
                arguments["coordinate2"], self.screen_width, self.screen_height
            )

        await self.phone_input_channel.put(
            tool_call, key=device_key, async_op=True
        ).async_wait()
        screenshot = await self.phone_output_channel.get(
            key=device_key, async_op=True
        ).async_wait()
        return screenshot

    async def run_one_query(self, user_instruction: str) -> AgentLoopOutput:
        """Run one mobile agent query."""
        async with self.device_context() as device_idx:
            return await self._run_one_query(user_instruction, device_idx=device_idx)

    async def _run_one_query(
        self, user_instruction: str, device_idx: int
    ) -> AgentLoopOutput:
        device_key = self.make_device_key(device_idx)

        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": get_system_prompt()}],
            },
        ]

        # Get initial screenshot
        screenshot = await self.get_screenshot(device_key=device_key)
        self.log_info(f"Agent[device={device_key}] received initial screenshot")

        all_response_ids = []
        all_response_mask = []
        trace_prints = []

        history = []

        for step in range(self.max_steps):
            # Build user message with query and screenshot
            # delete_images_in_history(messages)

            # Build history string
            stage2_history = ""
            for idx, his in enumerate(history):
                stage2_history += (
                    f"Step {idx + 1}: {his.replace(chr(10), '').replace(chr(34), '')}; "
                )
            llm_user_query = f"The user query: {user_instruction}.\nTask progress (You have done the following operation on the current device): {stage2_history}.\n"

            messages = [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": get_system_prompt()}],
                },
            ]
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": llm_user_query},
                        screenshot,
                    ],
                },
            )

            # Generate response from VL model
            result = await self.vl_generate(messages)
            response_text = result["text"]
            output_ids = result.get("output_ids", [])

            # Extract thought from output
            thought = ""
            for line in response_text.split("\n"):
                if line.strip().startswith("Thought:"):
                    thought = line.replace("Thought:", "").strip()
                    break

            # Extract action description from output
            action_description = ""
            for line in response_text.split("\n"):
                if line.strip().startswith("Action:"):
                    action_description = line.replace("Action:", "").strip()
                    break

            self.log_info(
                f"Agent[device={device_key}] Step {step + 1}: "
                f"\033[32m{response_text}\033[0m"
            )
            trace_prints.append(f"Step {step + 1}: {response_text}")

            # Add assistant response to history
            # messages.append(
            #     {
            #         "role": "assistant",
            #         "content": [{"type": "text", "text": response_text}],
            #     }
            # )

            # Track response ids (mask=1 for model generated)
            all_response_ids.extend(output_ids)
            all_response_mask.extend([1] * len(output_ids))

            # Parse and execute tool call
            tool_call = parse_tool_call(response_text)
            if not tool_call:
                self.log_info("Error: No tool call found in response")
                break

            action_name = tool_call.get("arguments", {}).get("action", "")

            if action_description or thought:
                history.append(f"Thought: {thought}\nAction: {action_description}\n")
            else:
                history.append(f"Performed {action_name} action")

            if action_name == "terminate":
                status = tool_call.get("arguments", {}).get("status", "success")
                self.log_info(f"\n{'=' * 60}")
                self.log_info(f"Task terminated with status: {status}")
                self.log_info(f"{'=' * 60}\n")
                break

            # Execute action and get new screenshot
            screenshot = await self.execute_action(tool_call, device_key=device_key)

        # Build prompt from initial messages (system + first user)
        prompt_text = self.tokenizer.apply_chat_template(
            messages[:2], tokenize=False, add_generation_prompt=True
        )
        prompt_ids_out = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        return AgentLoopOutput(
            prompt_ids=prompt_ids_out,
            response_ids=all_response_ids,
            ###
            ### WARN
            ## prompt_text=prompt_text,
            prompt_text=user_instruction,
            response_text="".join(
                [m["content"][0]["text"] for m in messages if m["role"] == "assistant"]
            ),
            response_mask=all_response_mask,
            num_turns=len([m for m in messages if m["role"] == "assistant"]),
            trace_prints=trace_prints,
        )

    async def stop_channels(self, device_keys: Optional[list[int]] = None):
        """Send termination signal to connected workers.

        Args:
            device_keys: List of local device indices to terminate.
                        If None, terminates all devices.
        """
        if device_keys is None:
            device_keys = list(range(self._num_devices))

        for device_idx in device_keys:
            key = self.make_device_key(device_idx)
            await self.phone_input_channel.put(
                None, key=key, async_op=True
            ).async_wait()

        await self.generate_input_channel.put(None, async_op=True).async_wait()

    # Keep the sync version for backward compatibility
    def run_one_test_query(
        self,
        channel_a2p: Channel,
        channel_p2a: Channel,
        channel_a2l: Channel,
        channel_l2a: Channel,
    ):
        """Synchronous test query for backward compatibility."""
        screen_width, screen_height = self.screen_width, self.screen_height
        max_steps = self.max_steps

        all_history = []

        user_query = "Set a alarm clock with the time 23:00 today."
        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": get_system_prompt()},
                ],
            },
        ]

        channel_a2p.put({"name": "screenshot"})
        screenshot = channel_p2a.get()
        self.log_info("agent get screenshot")

        for _ in range(max_steps):
            delete_images_in_history(messages)
            messages.append(
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_query},
                        screenshot,
                    ],
                },
            )

            channel_a2l.put(messages)
            response = channel_l2a.get()
            response_text = response[0]["text"]

            self.log_info(f"Response: \033[32m{response_text}\033[0m")

            messages.append(
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": response_text},
                    ],
                }
            )

            tool_call = parse_tool_call(response_text)

            if not tool_call:
                self.log_info("Error: No tool call found in response")
                return all_history

            action_name = tool_call.get("arguments", {}).get("action", "")
            print(f"Executing action: {action_name}")

            arguments = tool_call.get("arguments", {})
            if "coordinate" in arguments:
                arguments["coordinate"] = rescale_coordinates(
                    arguments["coordinate"], screen_width, screen_height
                )
                print(f"Rescaled coordinates: {arguments['coordinate']}")

            if "coordinate2" in arguments:
                arguments["coordinate2"] = rescale_coordinates(
                    arguments["coordinate2"], screen_width, screen_height
                )
                print(f"Rescaled coordinates2: {arguments['coordinate2']}")

            channel_a2p.put(tool_call)
            screenshot = channel_p2a.get()

            if action_name == "terminate":
                status = arguments.get("status", "success")
                self.log_info(f"\n{'=' * 60}")
                self.log_info(f"Task terminated with status: {status}")
                self.log_info(f"{'=' * 60}\n")
                all_history.append(copy.deepcopy(messages))
                break

            all_history.append(copy.deepcopy(messages))

        channel_a2p.put(None)
        channel_a2l.put(None)
        return all_history
