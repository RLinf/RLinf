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

"""Psi0 embodied policy integration."""

from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """Load a Psi0 checkpoint and wrap it as an RLinf policy."""
    from pathlib import Path

    from psi.config.config import LaunchConfig
    from psi.models.psi0 import Psi0Model
    from psi.utils import parse_args_to_tyro_config, seed_everything

    from rlinf.models.embodiment.psi0.psi0_action_model import Psi0Policy
    from rlinf.models.embodiment.psi0.psi0_processing import Psi0ProcessorAdapter

    run_dir = Path(str(cfg.model_path)).expanduser()
    checkpoint_step = cfg.get("checkpoint_step", 40000)
    if not run_dir.exists():
        raise ValueError(f"Psi0 run directory does not exist: {run_dir}.")

    config_template: LaunchConfig = parse_args_to_tyro_config(run_dir / "argv.txt")
    launch_config = config_template.model_validate_json(
        (run_dir / "run_config.json").read_text()
    )
    seed_everything(launch_config.seed or 42)
    model = Psi0Model.from_pretrained(
        run_dir,
        checkpoint_step,
        launch_config,
        device="cpu",
    )
    processor = Psi0ProcessorAdapter.from_upstream_transform(
        launch_config.data.transform.field,
        launch_config.data.transform.model,
    )
    psi0_cfg = cfg.psi0
    policy = Psi0Policy(
        model=model,
        processor=processor,
        num_inference_steps=int(psi0_cfg.num_inference_steps),
        plan_horizon=int(psi0_cfg.plan_horizon),
        execution_horizon=int(cfg.num_action_chunks),
        overlap_horizon=int(psi0_cfg.overlap_horizon),
        rtc_max_delay=int(launch_config.model.max_delay),
        stochastic_noise_scale=float(psi0_cfg.get("stochastic_noise_scale", 1.0)),
        add_value_head=bool(cfg.get("add_value_head", False)),
    )
    if hasattr(policy, "value_head") and torch_dtype is not None:
        policy.to(dtype=torch_dtype)
    return policy


__all__ = ["get_model"]
