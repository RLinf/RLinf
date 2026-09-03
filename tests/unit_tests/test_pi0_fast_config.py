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

import os
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf


def _load_config(*parts: str):
    repo_root = Path(__file__).resolve().parents[2]
    return OmegaConf.load(repo_root.joinpath(*parts))


def _load_pi0_fast_grpo_config():
    return _load_config(
        "examples",
        "embodiment",
        "config",
        "libero_10_grpo_pi0_fast.yaml",
    )


def _compose_example_config(config_name: str):
    repo_root = Path(__file__).resolve().parents[2]
    config_dir = repo_root / "examples" / "embodiment" / "config"
    old_embodied_path = os.environ.get("EMBODIED_PATH")
    os.environ["EMBODIED_PATH"] = str(config_dir.parent)
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(config_dir),
            job_name="pi0-fast-config-test",
        ):
            cfg = compose(config_name=config_name)
            OmegaConf.resolve(cfg)
            return cfg
    finally:
        if old_embodied_path is None:
            os.environ.pop("EMBODIED_PATH", None)
        else:
            os.environ["EMBODIED_PATH"] = old_embodied_path


def test_pi0_fast_grpo_clamps_log_ratio_before_exponentiation():
    cfg = _load_pi0_fast_grpo_config()

    assert cfg.algorithm.clip_log_ratio_min == -20
    assert cfg.algorithm.clip_log_ratio_max == 20


def test_pi0_fast_fsdp_uses_fp32_master_without_changing_compute_dtypes():
    configs = (
        _load_pi0_fast_grpo_config(),
        _load_config(
            "tests",
            "e2e_tests",
            "embodied",
            "libero_10_grpo_pi0_fast.yaml",
        ),
    )

    for cfg in configs:
        mixed_precision = cfg.actor.fsdp_config.mixed_precision
        assert mixed_precision.param_dtype is None
        assert mixed_precision.reduce_dtype is None
        assert mixed_precision.buffer_dtype is None
        assert cfg.actor.optim.use_fp32_master_params is True


def test_pi0_fast_grpo_uses_all_linear_lora():
    cfg = _load_pi0_fast_grpo_config()

    assert cfg.actor.model.is_lora is True
    assert cfg.actor.model.lora_rank == 16
    assert cfg.actor.model.lora_target_scope == "all_linear"
    assert cfg.rollout.model.pi0_fast.gradient_checkpointing is False


def test_pi0_fast_config_uses_explicit_local_artifact_paths():
    cfg = _load_config("examples", "embodiment", "config", "model", "pi0_fast.yaml")

    assert cfg.model_path == "/path/to/pi0fast-libero"
    assert cfg.pi0_fast.text_tokenizer_name == "/path/to/paligemma-3b-pt-224"
    assert cfg.pi0_fast.action_tokenizer_name == "/path/to/tokenizer-lib-mean"


def test_pi0_fast_published_experiment_sizes_match_validated_runs():
    eval_cfg = _load_config(
        "examples", "embodiment", "config", "libero_10_eval_pi0_fast.yaml"
    )
    train_cfg = _load_pi0_fast_grpo_config()

    assert eval_cfg.env.eval.total_num_envs == 500
    assert eval_cfg.rollout.sampling_params.temperature_eval == 0.0
    assert train_cfg.rollout.sampling_params.temperature_train == 0.3
    assert train_cfg.rollout.sampling_params.temperature_eval == 0.0
    assert train_cfg.runner.max_steps == 330
    assert train_cfg.runner.val_check_interval == 10
    assert train_cfg.env.train.total_num_envs == 256
    assert train_cfg.env.train.rollout_epoch == 4
    assert train_cfg.env.eval.total_num_envs == 256
    assert train_cfg.algorithm.group_size == 8
    assert train_cfg.actor.micro_batch_size == 16
    assert train_cfg.actor.global_batch_size == 13312
    assert train_cfg.actor.seed == 1234
    assert train_cfg.env.train.seed == 0


def test_pi0_fast_example_configs_compose_with_model_defaults():
    eval_cfg = _compose_example_config("libero_10_eval_pi0_fast")
    train_cfg = _compose_example_config("libero_10_grpo_pi0_fast")

    for cfg in (eval_cfg, train_cfg):
        model_cfg = cfg.rollout.model if cfg.runner.only_eval else cfg.actor.model
        assert model_cfg.model_type == "pi0_fast"
        assert model_cfg.model_path == "/path/to/pi0fast-libero"
        assert model_cfg.get("add_value_head") is None

    assert train_cfg.algorithm.adv_type == "grpo"
    assert train_cfg.algorithm.logprob_type == "sequence_token_level"
    assert train_cfg.algorithm.loss_agg_func == "token-mean"
    assert train_cfg.env.eval.seed == 0
    assert train_cfg.rollout.sampling_params.temperature_train == 0.3
    assert train_cfg.rollout.sampling_params.temperature_eval == 0.0
    assert "temperature_train" not in train_cfg.actor.model.pi0_fast
    assert "temperature_eval" not in train_cfg.actor.model.pi0_fast


def test_existing_starvla_config_still_composes():
    cfg = _compose_example_config("libero_spatial_grpo_starvla")

    assert cfg.actor.model.model_type == "starvla"
    assert cfg.algorithm.logprob_type == "chunk_level"
