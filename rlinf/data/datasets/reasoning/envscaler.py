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

import json
from pathlib import Path
from typing import Any

from omegaconf import DictConfig
from transformers import PreTrainedTokenizerBase

from rlinf.data.datasets.reasoning.dataset import ReasoningDataset


class EnvScalerDataset(ReasoningDataset):
    """Send public environment instructions and a scenario reference to the agent."""

    def __init__(
        self,
        data_paths: str | list[str],
        config: DictConfig,
        tokenizer: PreTrainedTokenizerBase,
    ):
        self.metadata_path = config.data.environment_path
        super().__init__(data_paths, config, tokenizer)

    def _load_data(self) -> list[dict[str, Any]]:
        with open(self.metadata_path, encoding="utf-8") as source:
            environments = json.load(source)
        instructions = {
            key: "\n".join([env["environment_introduction"], *env["constraints_rules"]])
            for key, env in environments.items()
        }
        rows = []
        for path in self.data_paths:
            path = str(Path(path).resolve())
            with open(path, encoding="utf-8") as source:
                tasks = json.load(source)
            for index, task in enumerate(tasks):
                rows.append(
                    {
                        self.prompt_key: instructions[task["env_id"]],
                        self.answer_key: {
                            "path": path,
                            "index": index,
                            "task_id": task["task_id"],
                            "env_id": task["env_id"],
                        },
                    }
                )
        return rows
