# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Per-task model builders for the OpenPI 0.5 factory.

:func:`rlinf.models.embodiment.openpi_pytorch.get_model` loads the vendored
``Pi0`` core and then dispatches on ``actor.model.openpi.task`` to one of the
builders here, each of which wraps the core in the concrete variant:

* :func:`_build_eval_model` → :class:`OpenPiPytorchEvalActionModel`
* :func:`_build_sft_model`  → :class:`OpenPiPytorchSFTActionModel`
* :func:`_build_rl_model`   → :class:`OpenPiPytorchRLActionModel`

The eval / RL builders assemble the shared ``openpi.transforms`` pipeline via
:func:`transforms_pipeline.build_openpi_transforms`; the SFT builder holds no
transforms (the SFT data loader applies them upstream).
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def _resolve_data_kwargs(cfg):
    """Read the optional ``openpi_data`` override block as a plain dict."""
    from omegaconf import OmegaConf

    data_kwargs = OmegaConf.select(cfg, "openpi_data", default=None)
    if data_kwargs is not None:
        data_kwargs = OmegaConf.to_container(data_kwargs, resolve=True)
    return data_kwargs


def _build_eval_model(
    cfg,
    model_cfg,
    model,
    *,
    num_steps,
    action_chunk,
    action_env_dim,
):
    """Build the eval variant: openpi.transforms pipeline, no value head, no
    chain collection, no training-mode forward.

    Produces a bare :class:`OpenPiPytorchEvalActionModel` — same eval path the
    RL subclass uses, but stripped of every PPO-only bit. Selected by
    ``actor.model.openpi.task: eval``.
    """
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.eval_action_model import (
        OpenPiPytorchEvalActionModel,
    )
    from rlinf.models.embodiment.openpi_pytorch.transforms_pipeline import (
        build_openpi_transforms,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi.config_name is required for task='eval' "
            "(it selects the upstream openpi TrainConfig, e.g. 'pi05_behavior')."
        )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )

    eval_model = OpenPiPytorchEvalActionModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
        action_chunk=action_chunk,
        config_name=config_name,
        state_indices=OmegaConf.select(model_cfg, "state_indices", default=None),
    )
    eval_model.setup_wrappers(input_transforms, output_transforms)
    return eval_model


def _build_sft_model(
    model,
    *,
    num_steps,
    action_env_dim,
):
    """Build the SFT variant.

    The observation/action transform is applied upstream in the environment SFT
    data loader, which routes each frame through the same openpi transform
    pipeline the eval/RL paths use, so the SFT model holds no processor and no
    transforms — it just computes the flow-matching loss.
    """
    from rlinf.models.embodiment.openpi_pytorch.sft_action_model import (
        OpenPiPytorchSFTActionModel,
    )

    return OpenPiPytorchSFTActionModel(
        model,
        num_steps=num_steps,
        action_env_dim=action_env_dim,
    )


def _build_rl_model(
    cfg,
    model_cfg,
    model,
    *,
    num_steps,
    action_chunk,
    action_env_dim,
    paligemma_width,
):
    """Build the RL variant: openpi.transforms pipeline + value head + chain
    sampler + optional train-expert-only freeze.

    Uses :func:`build_openpi_transforms` to derive the shared transforms
    pipeline (same logic the eval task path runs) and layers the PPO-only
    knobs (``rl_cfg``, value head, freeze) on top.
    """
    from omegaconf import OmegaConf

    from rlinf.models.embodiment.openpi_pytorch.rl_action_model import (
        OpenPiPytorchRLActionModel,
        OpenPiPytorchRLConfig,
    )
    from rlinf.models.embodiment.openpi_pytorch.transforms_pipeline import (
        build_openpi_transforms,
    )

    config_name = str(OmegaConf.select(model_cfg, "config_name", default=""))
    if not config_name:
        raise ValueError(
            "actor.model.openpi.config_name is required for task='rl' "
            "(it selects the upstream openpi TrainConfig, e.g. 'pi05_behavior')."
        )

    input_transforms, output_transforms = build_openpi_transforms(
        cfg.model_path, config_name, data_kwargs=_resolve_data_kwargs(cfg)
    )

    rl_cfg = OpenPiPytorchRLConfig(
        add_value_head=bool(OmegaConf.select(cfg, "add_value_head", default=False)),
        noise_method=str(
            OmegaConf.select(model_cfg, "noise_method", default="flow_ode")
        ),
        noise_level=float(OmegaConf.select(model_cfg, "noise_level", default=0.0)),
        joint_logprob=bool(OmegaConf.select(model_cfg, "joint_logprob", default=False)),
        ignore_last=bool(OmegaConf.select(model_cfg, "ignore_last", default=False)),
        value_after_vlm=bool(
            OmegaConf.select(model_cfg, "value_after_vlm", default=False)
        ),
        value_vlm_mode=str(
            OmegaConf.select(model_cfg, "value_vlm_mode", default="mean_token")
        ),
        detach_critic_input=bool(
            OmegaConf.select(model_cfg, "detach_critic_input", default=False)
        ),
        train_expert_only=bool(
            OmegaConf.select(model_cfg, "train_expert_only", default=False)
        ),
        config_name=config_name,
    )

    rl_model = OpenPiPytorchRLActionModel(
        model,
        num_steps=num_steps,
        action_chunk=action_chunk,
        action_env_dim=action_env_dim,
        rl_cfg=rl_cfg,
        paligemma_width=paligemma_width,
    )
    rl_model.setup_wrappers(input_transforms, output_transforms)
    if bool(OmegaConf.select(model_cfg, "train_expert_only", default=False)):
        # Mirror the legacy ``openpi/openpi_action_model.OpenPi0ForRLActionPrediction``
        # PPO path: freeze the PaliGemma VLM (SigLIP vision + LLM expert 0) and
        # only update the action expert + projections + value head. With the VLM
        # frozen, the autograd graph dead-ends at the prefix output, so PPO
        # backward never traverses the 2.5B paligemma — this is the dominant
        # ``actor/run_training`` parity lever vs the legacy implementation.
        frozen = rl_model.freeze_vlm()
        logger.info(
            "openpi_pytorch[rl]: train_expert_only=True; froze %d parameter tensors "
            "(SigLIP + gemma expert-0)",
            frozen,
        )
    return rl_model
