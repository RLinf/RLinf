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
from omegaconf import OmegaConf

from rlinf.envs.calvin.utils import get_sequences

ENV_CFG_DIR = Path(__file__).parent / "calvin_cfg/"


def _get_calvin_tasks_and_reward(num_sequences, task_suite_name="calvin_d"):
    conf_dir = (
        pathlib.Path(calvin_env.__file__).absolute().parents[2]
        / "calvin_models"
        / "conf"
    )
    if not conf_dir.exists():
        raise FileNotFoundError(
            f"Configuration directory {conf_dir} does not exist. "
            "Please ensure that the calvin_models package is installed correctly."
        )
    task_cfg_path = conf_dir / "callbacks/rollout/tasks/new_playtable_tasks.yaml"
    val_annotations_path_val = conf_dir / "annotations/new_playtable_validation.yaml"
    val_annotations_path_train = conf_dir / "annotations/new_playtable.yaml"
    if task_suite_name == "calvin_d":
        val_annotations = OmegaConf.load(val_annotations_path_val)
    elif task_suite_name == "calvin_abc":
        val_annotations = OmegaConf.load(val_annotations_path_train)
    elif task_suite_name == "calvin_abcd":
        ann_val = OmegaConf.load(val_annotations_path_val)
        ann_train = OmegaConf.load(val_annotations_path_train)
        val_annotations = OmegaConf.to_container(
            OmegaConf.merge(ann_val, ann_train), resolve=True
        )
    else:
        raise NotImplementedError(f"task suite {task_suite_name} is not yet supported.")

    task_cfg = OmegaConf.load(task_cfg_path)
    task_oracle = hydra.utils.instantiate(task_cfg)
    eval_sequences = get_sequences(num_sequences)
    return eval_sequences, val_annotations, task_oracle


def make_env(**kwargs):
    # Get current file directory
    if not ENV_CFG_DIR.exists():
        raise FileNotFoundError(
            f"Environment configuration directory {ENV_CFG_DIR} does not exist. "
            "Please ensure that the calvin_env package is installed correctly."
        )
    render_conf = OmegaConf.load(Path(ENV_CFG_DIR) / ".hydra" / "merged_config.yaml")
    if "scene" in kwargs:
        scene_cfg = OmegaConf.load(
            Path(ENV_CFG_DIR) / ".hydra" / f"{kwargs['scene']}.yaml"
        )
        render_conf.scene = scene_cfg
    env = hydra.utils.instantiate(
        render_conf.env, show_gui=False, use_vr=False, use_scene_info=True
    )
    return env


class CalvinBenchmark:
    def __init__(self, task_suite_name, _generator):
        assert task_suite_name in ["calvin_d", "calvin_abc", "calvin_abcd"], (
            f"task suite {self.cfg.task_suite_name} is not yet supported."
        )
        self.task_suite_name = task_suite_name
        self._generator = _generator
        self.eval_sequences, self.val_annotations, self.task_oracle = (
            _get_calvin_tasks_and_reward(
                self.get_num_tasks() * self.get_task_num_trials(),
                task_suite_name,
            )
        )

    def get_num_tasks(self):
        return 1

    def get_task_num_trials(self):
        return 1000

    def get_task_init_states(self, trial_id):
        return self.eval_sequences[trial_id][0]

    def get_task_sequence(self, trial_id):
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
        if self.task_suite_name == "calvin_d":
            # calvin_d using only the validation set(from annotations/new_playtable_validation.yaml), which has only one prompt for each task;
            return self.val_annotations[task][0]
        else:
            task_descriptions = self.val_annotations[task]
            prompt_id = int(
                self._generator.integers(low=0, high=len(task_descriptions))
            )
            return task_descriptions[prompt_id]

    def check_subtask_success(self, prev_info, current_info, subtask):
        current_task_info = self.task_oracle.get_task_info_for_set(
            prev_info, current_info, {subtask}
        )
        return len(current_task_info) > 0
