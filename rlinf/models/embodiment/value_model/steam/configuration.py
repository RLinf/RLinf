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

"""STEAM value model configuration.

Shares the SigLIP + Gemma3 backbone knobs with the sibling embodied value
models, but:

* the categorical-value fields (``num_bins`` / ``bin_min`` / ``bin_max``) are
  replaced by ``label_smoothing`` because the head is a single-logit binary
  classifier, and
* ``action_dim`` / ``action_horizon`` are dropped because a ``(frame_t,
  frame_{t+k})`` pair model doesn't consume actions.

A new ``num_frames_per_pair`` knob (default 2) controls the image-token
fanout in ``modeling_steam.SteamBackbone``.
"""

from typing import Optional

from transformers import PretrainedConfig


def validate_steam_ensemble_settings(
    *,
    ensemble_size: int,
    micro_batch_size: Optional[int] = None,
    global_batch_size: Optional[int] = None,
) -> int:
    """Validate the ensemble size shared by training and inference.

    The ensemble always reduces members with the worst-of-N (minimum) rule
    from the STEAM paper, so there is no inference-mode knob to validate.

    ``micro_batch_size`` / ``global_batch_size`` are accepted for call-site
    parity but are intentionally not constrained: per-member sequential
    training treats both as the per-member batch (each member fetches its
    own micro batches independently), so divisibility by ``ensemble_size``
    is no longer required.
    """
    del micro_batch_size, global_batch_size  # accepted for backward compat

    ensemble_size = int(ensemble_size)
    if ensemble_size < 1:
        raise ValueError("ensemble_size must be >= 1")

    return ensemble_size


class SteamConfig(PretrainedConfig):
    """Configuration for the :class:`SteamCriticModel`.

    Uses a SigLIP vision encoder (default ``google/siglip-so400m-patch14-384``,
    native 384x384) and a Gemma3-270m language backbone, fused via
    per-frame-concat + 2-layer MLP, with a scalar binary logit head.
    """

    model_type = "steam_value_model"

    def __init__(
        self,
        # Backbones
        vision_repo_id: str = "",
        language_repo_id: str = "",
        vision_revision: Optional[str] = None,
        language_revision: Optional[str] = None,
        # Fusion + binary / multi-bin head
        fusion_hidden_dim: int = 512,
        dropout: float = 0.1,
        label_smoothing: float = 0.05,
        num_frames_per_pair: int = 2,
        # num_bins == 2 → legacy binary mode (fixed k, labels are long
        # bin indices in {0, 1}: 0 = regress, 1 = progress).
        # num_bins  > 2 → multi-bin mode: pair_dataset samples i ∈ [1, K],
        # signed stride in [-K, K] \ {0} is discretized into num_bins
        # contiguous bins. Must be even so the sign split lands exactly at
        # num_bins // 2.
        num_bins: int = 2,
        stride_k: Optional[int] = None,
        ensemble_size: int = 1,
        ensemble_head_seed_base: Optional[int] = None,
        # Runtime
        dtype: str = "bfloat16",
        precision: Optional[str] = None,
        freeze_vision_encoder: bool = False,
        freeze_language_model: bool = True,
        use_gradient_checkpointing: bool = False,
        # Prompt tokenization length (SteamProcessor interface compat).
        max_token_len: int = 200,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.vision_repo_id = vision_repo_id
        self.language_repo_id = language_repo_id
        self.vision_revision = vision_revision
        self.language_revision = language_revision

        self.fusion_hidden_dim = fusion_hidden_dim
        self.dropout = dropout
        self.label_smoothing = label_smoothing
        self.num_frames_per_pair = num_frames_per_pair
        self.num_bins = int(num_bins)
        self.stride_k = None if stride_k is None else int(stride_k)
        self.ensemble_size = int(ensemble_size)
        self.ensemble_head_seed_base = (
            None if ensemble_head_seed_base is None else int(ensemble_head_seed_base)
        )

        self.dtype = precision if precision is not None else dtype
        self.precision = self.dtype
        self.freeze_vision_encoder = freeze_vision_encoder
        self.freeze_language_model = freeze_language_model
        self.use_gradient_checkpointing = use_gradient_checkpointing

        self.max_token_len = max_token_len

        self._validate()

    def _validate(self) -> None:
        if not self.vision_repo_id:
            raise ValueError(
                "SteamConfig.vision_repo_id must be a non-empty path or HF repo id"
            )
        if not self.language_repo_id:
            raise ValueError(
                "SteamConfig.language_repo_id must be a non-empty path or HF repo id"
            )
        if self.fusion_hidden_dim <= 0:
            raise ValueError("fusion_hidden_dim must be > 0")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(f"dropout must be in [0, 1), got {self.dropout}")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError(
                f"label_smoothing must be in [0, 1), got {self.label_smoothing}"
            )
        if self.num_frames_per_pair < 1:
            raise ValueError("num_frames_per_pair must be >= 1")
        # num_bins must be even so the progressive / regressive split lands
        # exactly at num_bins // 2. num_bins == 2 selects the legacy binary
        # mode; num_bins > 2 activates multi-bin. An odd num_bins would leave
        # a center bin straddling signed-stride == 0, which we don't sample.
        if self.num_bins < 2 or self.num_bins % 2 != 0:
            raise ValueError(f"num_bins must be >= 2 and even, got {self.num_bins}")
        if self.ensemble_size < 1:
            raise ValueError("ensemble_size must be >= 1")
        if self.dtype not in {"bfloat16", "float32", "float16"}:
            raise ValueError(
                f"dtype must be one of bfloat16/float32/float16, got {self.dtype}"
            )
        if self.max_token_len <= 0:
            raise ValueError("max_token_len must be > 0")

    def to_diff_dict(self) -> dict:
        """Return a full config dict without instantiating an empty default config.

        ``PretrainedConfig.to_diff_dict`` creates ``self.__class__()`` to compute
        a diff against default values. That does not work here because
        ``SteamConfig`` intentionally requires non-empty backbone ids.
        Returning the full config keeps HuggingFace save/load helpers working
        for checkpoint metadata.
        """
        return self.to_dict()
