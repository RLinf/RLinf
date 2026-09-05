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

import torch
from omegaconf import OmegaConf

from rlinf.envs.behavior.observation_runtime import (
    apply_trunk_proprio_values,
    compose_task_description,
    parse_trunk_proprio_randomization,
    sample_trunk_proprio_values,
    task_descriptions_from_infos,
    update_stage_prompts_from_info,
)


def test_parse_trunk_proprio_randomization_disabled():
    assert parse_trunk_proprio_randomization(OmegaConf.create({})) is None


def test_sample_and_apply_trunk_proprio_values_preserve_dtype_and_rows():
    cfg = {
        "indices": [0, 2],
        "low": [0.0, 10.0],
        "high": [1.0, 20.0],
        "fixed_values": [0.5, None],
    }
    generator = torch.Generator().manual_seed(123)
    values = sample_trunk_proprio_values(cfg, count=3, generator=generator)
    states = torch.zeros((2, 4), dtype=torch.float64)

    randomized = apply_trunk_proprio_values(
        states,
        config=cfg,
        values=values,
        env_indices=[2, 0],
    )

    assert randomized.dtype == torch.float64
    assert torch.equal(states, torch.zeros_like(states))
    assert randomized[0, 0].item() == 0.5
    assert randomized[1, 0].item() == 0.5
    assert randomized[0, 2].item() == values[2, 1].item()
    assert randomized[1, 2].item() == values[0, 1].item()


def test_apply_trunk_proprio_values_falls_back_to_compact_state_indices():
    cfg = {
        "indices": [236, 237, 238, 239],
        "low": [0.0, 0.0, 0.0, 0.0],
        "high": [1.0, 1.0, 1.0, 1.0],
        "fixed_values": [None, None, None, None],
    }
    values = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    states = torch.zeros((1, 7))

    randomized = apply_trunk_proprio_values(
        states,
        config=cfg,
        values=values,
        env_indices=[0],
    )

    assert torch.equal(randomized[0, 3:7], values[0])


def test_stage_prompt_extraction_and_task_description_composition():
    stage_prompt_lists = [None]
    info = {
        "reward": {"task_specific": {"current_stage_idx": 1}},
        "replay_init": {"replay_stage_prompts": ["move", "press"]},
    }

    prompt = update_stage_prompts_from_info(stage_prompt_lists, 0, info)

    assert prompt == "press"
    assert stage_prompt_lists == [["move", "press"]]
    assert compose_task_description(None, "turn on radio", prompt) == (
        "turn on radio\npress"
    )
    assert compose_task_description("override", "turn on radio", prompt) == "override"


def test_task_descriptions_from_infos_uses_override_first():
    descriptions = task_descriptions_from_infos(
        num_envs=2,
        prompt_override="press radio",
        use_subtask_prompt=True,
        task_description="turn on radio",
        stage_prompt_lists=[None, None],
        infos=[{}, {}],
    )

    assert descriptions == ["press radio", "press radio"]


def test_task_descriptions_from_infos_updates_global_rows_on_partial_reset():
    stage_prompt_lists = [None, None, None, None]
    infos = [
        {
            "reward": {"task_specific": {"current_stage_idx": 0}},
            "replay_init": {"replay_stage_prompts": ["row one"]},
        },
        {
            "reward": {"task_specific": {"current_stage_idx": 1}},
            "replay_init": {"replay_stage_prompts": ["first", "row three"]},
        },
    ]

    descriptions = task_descriptions_from_infos(
        num_envs=4,
        prompt_override=None,
        use_subtask_prompt=True,
        task_description="turn on radio",
        stage_prompt_lists=stage_prompt_lists,
        infos=infos,
        env_indices=[1, 3],
    )

    assert descriptions == ["turn on radio\nrow one", "turn on radio\nrow three"]
    assert stage_prompt_lists[0] is None
    assert stage_prompt_lists[1] == ["row one"]
    assert stage_prompt_lists[2] is None
    assert stage_prompt_lists[3] == ["first", "row three"]
