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
import os

from rlinf.envs.metaworld.utils import load_prompt_from_json


class MetaWorldBenchmark:
    def __init__(self, task_suite_name):
        assert task_suite_name in [
            "metaworld_50",
            "metaworld_45_ind",
            "metaworld_45_ood",
        ]
        self.task_suite_name = task_suite_name
        config_path = os.path.join(os.path.dirname(__file__), "metaworld_config.json")
        self.task_description_dict = load_prompt_from_json(
            config_path, "TASK_DESCRIPTIONS"
        )
        self.ML45_dict = load_prompt_from_json(config_path, "ML45")

    def get_num_tasks(self):
        if self.task_suite_name == "metaworld_50":
            return 50
        elif self.task_suite_name == "metaworld_45_ind":
            return 45
        elif self.task_suite_name == "metaworld_45_ood":
            return 5

    def get_task_num_trials(self):
        if self.task_suite_name == "metaworld_50":
            return 10
        elif self.task_suite_name == "metaworld_45_ind":
            return 10
        elif self.task_suite_name == "metaworld_45_ood":
            return 20

    def get_env_names(self):
        if self.task_suite_name == "metaworld_50":
            return list(self.task_description_dict.keys())
        elif self.task_suite_name == "metaworld_45_ind":
            return self.ML45_dict["train"]
        elif self.task_suite_name == "metaworld_45_ood":
            return self.ML45_dict["test"]

    def get_task_description(self):
        task_descriptions = []
        for env_name in self.get_env_names():
            task_descriptions.append(self.task_description_dict[env_name])
        return task_descriptions
