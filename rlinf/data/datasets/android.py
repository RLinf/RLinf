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

import logging
import random
import sys
from typing import Any, Optional

import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer

from rlinf.data.datasets.item import DatasetItem
from rlinf.data.utils import batch_pad_to_fixed_len

LOGGER = logging.getLogger(__name__)


class AndroidWorldDataset(Dataset):
    def __init__(
        self,
        config: DictConfig,
        tokenizer: AutoTokenizer,
        seed: Optional[int] = None,
    ) -> None:
        """Dataset for AndroidWorld tasks.

        Args:
            config: RLinf config, including data.task_family, data.n_instances_per_task,
                and other AndroidWorld-related fields.
            tokenizer: Tokenizer for encoding the text inputs.
            seed: Optional random seed for generating task instances.
        """
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.max_prompt_length = config.data.max_prompt_length

        # android world specific parameters
        self.seed = seed if seed is not None else config.data.get("seed", 42)
        self.task_family = config.data.get("task_family", "android_world")
        self.n_instances_per_task = config.data.get("n_instances_per_task", 1)
        self.task_name = config.data.get("task_name", None)
        self.max_complexity = config.data.get("max_complexity", None)
        # Chat template
        self.apply_chat_template = config.data.get("apply_chat_template", False)
        self.system_prompt = self._get_default_system_prompt()

        self.tasks = self._load_data()

        LOGGER.info(
            "AndroidWorldDataset: loaded %d task instances from family '%s'",
            len(self.tasks),
            self.task_family,
        )
        if config.data.get("filter_prompt_by_length", False):
            self._filter_by_length()

    def _get_default_system_prompt(self) -> str:
        return (
            "You are an agent who can operate an Android phone on behalf of a user."
            " Based on user's goal/request, you may\n"
            "- Answer back if the request/goal is a question (or a chat message),"
            ' like user asks "What is my schedule for today?".\n'
            "- Complete some tasks described in the requests/goals by"
            " performing actions (step by step) on the phone.\n\n"
            "When given a user request, you will try to complete it step by step."
            " At each step, you will be given the current screenshot (including the"
            " original screenshot and the same screenshot with bounding"
            " boxes and numeric indexes added to some UI elements) and a history of"
            " what you have done (in text). Based on these pieces of information and"
            " the goal, you must choose to perform one of the"
            " action in the following list (action description followed by the JSON"
            " format) by outputing the action in the correct JSON format.\n"
            "- If you think the task has been completed, finish the task by using the"
            " status action with complete as goal_status:"
            ' `{{"action_type": "status", "goal_status": "complete"}}`\n'
            "- If you think the task is not feasible (including cases like you don't"
            " have enough information or can not perform some necessary actions),"
            " finish by using the `status` action with infeasible as goal_status:"
            ' `{{"action_type": "status", "goal_status": "infeasible"}}`\n'
            "- Answer user's question:"
            ' `{{"action_type": "answer", "text": "<answer_text>"}}`\n'
            "- Click/tap on an element on the screen. We have added marks (bounding"
            " boxes with numeric indexes on their TOP LEFT corner) to most of the UI"
            " elements in the screenshot, use the numeric index to indicate which"
            " element you want to click:"
            ' `{{"action_type": "click", "index": <target_index>}}`.\n'
            "- Long press on an element on the screen, similar with the click action"
            " above,use the numeric label on the bounding box to indicate which"
            " element you want to long press:"
            ' `{{"action_type": "long_press", "index": <target_index>}}`.\n'
            "- Type text into a text field (this action contains clicking the text"
            " field, typing in the text and pressing the enter, so no need to click on"
            " the target field to start), use the numeric label"
            " on the bounding box to indicate the target text field:"
            ' `{{"action_type": "input_text", "text": <text_input>,'
            ' "index": <target_index>}}`\n'
            '- Press the Enter key: `{{"action_type": "keyboard_enter"}}`\n'
            '- Navigate to the home screen: `{{"action_type": "navigate_home"}}`\n'
            '- Navigate back: `{{"action_type": "navigate_back"}}`\n'
            "- Scroll the screen or a scrollable UI element in one of the four"
            " directions, use the same numeric index as above if you want to scroll a"
            " specific UI element, leave it empty when scroll the whole screen:"
            ' `{{"action_type": "scroll", "direction": <up, down, left, right>,'
            ' "index": <optional_target_index>}}`\n'
            "- Open an app (nothing will happen if the app is not"
            ' installed): `{{"action_type": "open_app", "app_name": <name>}}`\n'
            '- Wait for the screen to update: `{{"action_type": "wait"}}`\n'
        )

    def _load_data(self) -> list[dict[str, Any]]:
        """Load tasks from AndroidWorld registry."""
        android_world_parent = self.config.data.get("android_world_parent", None)
        if android_world_parent and android_world_parent not in sys.path:
            sys.path.insert(0, android_world_parent)
        try:
            from android_world.registry import TaskRegistry  # type: ignore[import]
        except ImportError as exc:
            raise ImportError(
                "Cannot import android_world.registry. Please ensure android_world "
                f"is installed or the path is correct. Expected path: {android_world_parent}"
            ) from exc
        registry = TaskRegistry()
        task_registry = registry.get_registry(self.task_family)
        if self.task_name is not None:
            task_registry = {
                name: cls
                for name, cls in task_registry.items()
                if name == self.task_name
            }
            if not task_registry:
                LOGGER.warning("Task name %s not found in registry.", self.task_name)

        tasks: list[dict[str, Any]] = []
        for task_name, task_class in task_registry.items():
            if self.max_complexity is not None:
                try:
                    if hasattr(task_class, "complexity"):
                        complexity = task_class.complexity
                        if complexity > self.max_complexity:
                            LOGGER.debug(
                                "Skip task %s, complexity %s > %s",
                                task_name,
                                complexity,
                                self.max_complexity,
                            )
                            continue
                except Exception:
                    pass
            for i in range(self.n_instances_per_task):
                try:
                    instance_seed = self.seed + hash(f"{task_name}_{i}") % (2**31)
                    random.seed(instance_seed)

                    params = task_class.generate_random_params()

                    task_instance = task_class(params)
                    prompt = task_instance.goal  # The language goal constructed from the template with the params.

                    tasks.append(
                        {
                            "prompt": prompt,
                            "task_class": task_class,
                            "task_instance": task_instance,
                            "task_name": task_name,
                            "params": params,
                            "instance_seed": instance_seed,
                            "complexity": getattr(task_class, "complexity", None),
                        }
                    )
                except Exception as exc:  # pylint: disable=broad-except
                    LOGGER.warning(
                        "Error generating task %s instance %d: %s",
                        task_name,
                        i,
                        exc,
                    )
                    continue
        return tasks

    def __len__(self) -> int:
        return len(self.tasks)

    def _filter_by_length(self) -> None:
        total = len(self.tasks)
        filtered = []
        failed = 0
        for task in self.tasks:
            try:
                formatted_prompt = self._format_prompt(task["prompt"])
                _, length = self.encode(formatted_prompt)
                if length <= self.max_prompt_length:
                    filtered.append(task)
            except Exception:
                failed += 1
        self.tasks = filtered
        if len(self.tasks) == 0:
            raise ValueError(
                f"No tasks with prompt length less than max_prompt_length={self.max_prompt_length}."
                "Please check the configuration or increase max_prompt_length."
            )

        if failed > 0:
            LOGGER.warning(
                "%d tasks were skipped due to format issues (kept %d / %d)",
                failed,
                len(self.tasks),
                total,
            )

    def encode(self, text: str) -> tuple[list[int], int]:
        """Encode text and return token ids and length."""
        text_ids = self.tokenizer.encode(text)
        return text_ids, len(text_ids)

    def _format_prompt(self, user_prompt: str) -> str:
        """Format the prompt, optionally applying chat template.

        Args:
            user_prompt: User task description (``task.goal``).

        Returns:
            Formatted prompt text.
        """
        if self.apply_chat_template:
            messages = []
            if self.system_prompt:
                messages.append(
                    {
                        "role": "system",
                        "content": self.system_prompt,
                    }
                )

            messages.append(
                {
                    "role": "user",
                    "content": user_prompt,
                }
            )

            try:
                formatted = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                return formatted
            except Exception as exc:  # pylint: disable=broad-except
                LOGGER.warning(
                    "Failed to apply chat template: %s, using original prompt", exc
                )
                if self.system_prompt:
                    return f"{self.system_prompt}\n\nUser: {user_prompt}\nAssistant:"
                return user_prompt
        else:
            if self.system_prompt:
                return f"{self.system_prompt}\n\nUser: {user_prompt}\nAssistant:"
            return user_prompt

    def __getitem__(self, idx: int) -> DatasetItem:
        """Return one encoded AndroidWorld task instance."""
        task = self.tasks[idx]
        user_prompt = task["prompt"]
        prompt_text = self._format_prompt(user_prompt)
        prompt_tokens, prompt_length = self.encode(prompt_text)
        prompt_tokens_tensor = torch.as_tensor(prompt_tokens, dtype=torch.int64)
        if prompt_length > self.max_prompt_length:
            LOGGER.warning(
                "Prompt length %d exceeds max_prompt_length %d; truncating.",
                prompt_length,
                self.max_prompt_length,
            )
            prompt_tokens_tensor = prompt_tokens_tensor[: self.max_prompt_length]
            prompt_length = self.max_prompt_length

        prompt_tokens_tensor = batch_pad_to_fixed_len(
            [prompt_tokens_tensor],
            self.max_prompt_length,
            self.tokenizer.eos_token_id,
            left_pad=True,
        )[0]

        # used for reward worker to reconstruct the task instance
        answer = {
            "task_name": task["task_name"],
            "params": task["params"],
            "instance_seed": task["instance_seed"],
            "class_name": task["task_class"].__name__,
            "task": task["task_instance"],
        }
        # extra metadata for the task
        meta_data = {
            "task_name": task["task_name"],
            "task_class_name": task["task_class"].__name__,
            "complexity": task["complexity"],
            "task_class": task["task_class"],
            "user_prompt": user_prompt,  # goal
            "system_prompt": self.system_prompt if self.system_prompt else None,
            "apply_chat_template": self.apply_chat_template,
        }

        return DatasetItem(
            prompt=prompt_tokens_tensor,
            length=prompt_length,
            answer=answer,
            idx=idx,
            prompt_text=prompt_text,
            meta=meta_data,
        )
