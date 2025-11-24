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
import pathlib
from pathlib import Path

import calvin_env
import hydra
from calvin_agent.evaluation.utils import get_env_state_for_initial_condition
from calvin_env.envs.play_table_env import get_env
from omegaconf import OmegaConf

from rlinf.envs.calvin.utils import get_sequences


def _get_calvin_tasks_and_reward(num_sequences, use_random_seed=True):
    conf_dir = (
        pathlib.Path(calvin_env.__file__).absolute().parents[1]
        / "calvin_agent"
        / "conf"
    )
    task_cfg = OmegaConf.load(
        conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    )
    task_oracle = hydra.utils.instantiate(task_cfg)
    val_annotations = OmegaConf.load(
        conf_dir / "annotations/new_playtable_validation.yaml"
    )
    eval_sequences = get_sequences(num_sequences, use_random_seed=use_random_seed)
    return eval_sequences, val_annotations, task_oracle


def make_env():
    dataset_paths = [
        "/mnt/mnt/public/chenkang/rlinf_pi/calvin/dataset/calvin_debug_dataset",
        "/mnt/mnt/public/chenkang/RLinf_fork_calvin/calvin_debug_dataset",  # Add your second path here
    ]
    for path in dataset_paths:
        try:
            return get_env(Path(path) / "validation", show_gui=False)
        except Exception:
            continue
    raise RuntimeError(f"Failed to create environment from all paths: {dataset_paths}")


class CalvinBenchmark:
    def __init__(self, task_suite_name):
        assert task_suite_name in [
            "calvin_d",
        ]
        self.task_suite_name = task_suite_name
        self.use_random_seed = True  # True for rollout and False for val
        self.eval_sequences, self.val_annotations, self.task_oracle = (
            _get_calvin_tasks_and_reward(
                self.get_num_tasks() * self.get_task_num_trials(), self.use_random_seed
            )
        )

    def get_num_tasks(self):
        if self.task_suite_name == "calvin_d":
            return 1

    def get_task_num_trials(self):
        if self.task_suite_name == "calvin_d":
            return 1000

    def get_task_init_states(self, trial_id):
        if self.task_suite_name == "calvin_d":
            return self.eval_sequences[trial_id][0]

    def get_task_sequence(self, trial_id):
        if self.task_suite_name == "calvin_d":
            return self.eval_sequences[trial_id][1]

    def get_obs_for_initial_condition(self, init_states):
        robot_obs_list = []
        scene_obs_list = []
        for idx in range(len(init_states)):
            robot_obs, scene_obs = get_env_state_for_initial_condition(init_states[idx])
            robot_obs_list.append(robot_obs)
            scene_obs_list.append(scene_obs)
        return robot_obs_list, scene_obs_list

    def get_task_descriptions(self, task):
        return self.val_annotations[task][0]

    def check_subtask_success(self, prev_info, current_info, subtask):
        current_task_info = self.task_oracle.get_task_info_for_set(
            prev_info, current_info, {subtask}
        )
        return len(current_task_info) > 0
