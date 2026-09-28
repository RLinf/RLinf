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

"""Streaming Flow Policy sidecar config used by SFT and eval ``Pi0``.

Not a ``tasks/`` entry: YAML is still ``task: sft`` or ``task: eval`` plus
``use_sfp``, the same way ``use_rlt`` selects the RLT objective.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from rlinf.models.embodiment.openpi.rlt_config import OpenPiPytorchRLTConfig


@dataclasses.dataclass(frozen=True)
class OpenPiPytorchSfpConfig:
    """Selects the SFP objective and its noise schedule.

    ``use_sfp`` picks the SFP objective for SFT and the trajectory sampler for
    eval. The parameters, their shapes, and the checkpoint layout stay those of
    an ordinary Pi0.5, so an SFP run starts from a stock Pi0.5 checkpoint and
    its output loads back into one.
    """

    use_sfp: bool = False
    sigma: float = 0.16
    noise_decay: float = 4.0


def build_sfp_config(model_cfg: Any) -> OpenPiPytorchSfpConfig:
    """Build the optional SFP config from ``actor.model.openpi``."""
    from omegaconf import OmegaConf

    return OpenPiPytorchSfpConfig(
        use_sfp=bool(OmegaConf.select(model_cfg, "use_sfp", default=False)),
        sigma=float(OmegaConf.select(model_cfg, "sfp_sigma", default=0.16)),
        noise_decay=float(OmegaConf.select(model_cfg, "sfp_noise_decay", default=4.0)),
    )


def validate_sfp_config(
    sfp_cfg: OpenPiPytorchSfpConfig,
    rlt_cfg: OpenPiPytorchRLTConfig,
    task: str,
    *,
    pi05: bool,
) -> None:
    """Reject the configurations SFP cannot run under.

    SFP sends one suffix token and conditions Pi0.5 with adaRMS. Pi0 instead
    concatenates the time embedding onto every action token, so a one-token
    suffix does not match that layout. RL, DAgger, and DSRL sample actions
    with the flow-matching sampler, which integrates the wrong field on an SFP
    checkpoint. Eval has its own trajectory sampler. The RLT objective extends
    flow matching rather than SFP, so the two cannot be combined either.
    """
    if not sfp_cfg.use_sfp:
        return
    if not pi05:
        raise ValueError(
            "actor.model.openpi.use_sfp requires pi05=true. Streaming Flow "
            "Policy uses one suffix token and Pi0.5 adaRMS time conditioning. "
            "Pi0 mixes the time embedding into every action token and cannot "
            "represent that suffix."
        )
    if task not in ("sft", "eval"):
        raise ValueError(
            f"actor.model.openpi.use_sfp is not supported with task={task!r}: "
            "Streaming Flow Policy only trains under task='sft' and samples "
            "actions under task='eval'. RL, DAgger, and DSRL still need a "
            "trajectory logprob sampler."
        )
    if rlt_cfg.use_rlt:
        raise ValueError(
            "actor.model.openpi.use_sfp and use_rlt are mutually exclusive; "
            "the RLT objective extends flow matching, not SFP."
        )
