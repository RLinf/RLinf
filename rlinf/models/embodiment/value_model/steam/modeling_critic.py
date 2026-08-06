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

"""STEAM value critic — RLinf-facing entry point.

The observation/forward contract is compatible with the sibling
``rlinf.models.embodiment.value_model.recap.modeling_critic`` (same keys in
the observation dict, same ``images: dict[cam_name, Tensor[B,3,H,W]]``
layout, same CriticOutput dataclass), so the FSDP value-SFT worker and the
offline advantage pipeline can dispatch on ``model_type`` alone.

Compared with a categorical value-regression critic:
    * ``forward(observation, labels)`` replaces ``(observation, target_values)``
      — labels are long bin indices in ``[0, num_bins)`` (binary degenerates
      to ``0 = regress, 1 = progress``).
    * :meth:`_compute_loss` replaces ``_compute_categorical_loss`` with a
      ``num_bins``-way cross-entropy that covers both binary and multi-bin.
    * ``CriticOutput.predicted_values`` holds a signed, bin-weighted
      expectation in ``[-1, 1]`` — ``Σ_b p_b · signed_bin_b / half`` where
      ``signed_bin ∈ {-half, …, -1, 1, …, half}`` and ``half = num_bins //
      2`` — instead of a scalar expected value over value bins.
      ``CriticOutput.atoms`` is ``None``.

Public API (mirrors recap.modeling_critic.ValueCriticModel):
    - forward(observation, labels=None) -> CriticOutput
    - predict(observation) -> CriticOutput
    - predict_value(observation) -> Tensor   (signed value in [-1, 1])
    - from_checkpoint(checkpoint_dir, **kwargs) -> SteamCriticModel
    - gradient_checkpointing_enable() / .disable()
    - _no_split_modules / _no_split_names properties
"""

import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from transformers.modeling_outputs import ModelOutput

from .configuration import SteamConfig
from .modeling_steam import (
    SteamBackbone,
    _module_parameter_dtype,
)

if TYPE_CHECKING:
    # Type-check-only import to avoid a circular dependency: the ensemble
    # wrapper imports SteamCriticModel, so keep the back-reference
    # out of runtime.
    from .ensemble_modeling_critic import EnsembleSteamCriticModel

logger = logging.getLogger(__name__)


def _resolve_tokenizer_source(
    checkpoint_dir,
    explicit_tokenizer_path: Optional[str],
    model_config: SteamConfig,
) -> tuple[str, bool]:
    """Resolve the tokenizer source for inference-time checkpoint loading."""
    tokenizer_files = (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
    )

    if explicit_tokenizer_path:
        return explicit_tokenizer_path, os.path.exists(explicit_tokenizer_path)

    if any(
        os.path.exists(os.path.join(str(checkpoint_dir), name))
        for name in tokenizer_files
    ):
        return str(checkpoint_dir), True

    model_tokenizer_source = getattr(model_config, "language_repo_id", None)
    if model_tokenizer_source:
        return model_tokenizer_source, os.path.exists(model_tokenizer_source)

    raise ValueError(
        "No tokenizer found. Set tokenizer_path, save tokenizer files into the "
        f"checkpoint, or ensure the checkpoint config provides language_repo_id. "
        f"checkpoint_dir={checkpoint_dir}"
    )


@dataclass
class CriticOutput(ModelOutput):
    """Output for the single-model STEAM binary critic.

    Field list deliberately matches :class:`~rlinf.models.embodiment.value_model.recap.\
modeling_critic.CriticOutput` so worker code stays duck-type-compatible.
    For the binary variant:

        * ``logits`` is ``[B, num_bins]``.
        * ``probs`` is ``softmax(logits)`` — shape ``[B, num_bins]``.
        * ``predicted_values`` is a signed bin-weighted expectation in
          ``[-1, 1]`` — shape ``[B]`` (see
          :meth:`SteamCriticModel._predicted_signed_value`). For
          ``num_bins == 2`` this degenerates to ``2 · P(progress) - 1``.
        * ``atoms`` is always ``None``.
        * ``cat_acc_best`` carries binary accuracy for parity with the
          categorical critic's logging (the other cat_* fields stay
          ``None``).

    Ensemble aggregate fields (member-wise predictions, mean/min/variance)
    are deliberately **not** on this dataclass — they live on
    :class:`~rlinf.models.embodiment.value_model.steam.\
ensemble_modeling_critic.EnsembleCriticOutput`, which is what the
    :class:`EnsembleSteamCriticModel` wrapper returns. Consumers that
    need those stats must call ensemble-specific surfaces (see
    ``ensemble_modeling_critic.py``).
    """

    loss: Optional[torch.FloatTensor] = None
    predicted_values: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    probs: Optional[torch.FloatTensor] = None
    atoms: Optional[torch.FloatTensor] = None
    expert_loss: Optional[torch.FloatTensor] = None
    hidden_states: Optional[torch.FloatTensor] = None
    cat_acc_best: Optional[torch.FloatTensor] = None
    cat_acc_neighbor: Optional[torch.FloatTensor] = None
    mae: Optional[torch.FloatTensor] = None
    progress_values: Optional[torch.FloatTensor] = None


class SteamCriticModel(nn.Module):
    """STEAM value model — parallel to ValueCriticModel.

    Wraps the :class:`SteamBackbone` (SigLIP + Gemma3 + per-frame
    concat MLP) with the RLinf-side observation/forward contract:

        - ``forward(observation, labels)`` returns a :class:`CriticOutput`.
        - ``observation`` is a dict in the same format produced by
          :class:`~rlinf.models.embodiment.value_model.recap.data_collator.\
ValueDataCollator`: ``images: dict[cam_name, Tensor[B,3,H,W]]`` in [0, 1],
          ``image_masks: dict[cam_name, Tensor[B]]``, ``tokenized_prompt``,
          ``tokenized_prompt_mask``. The ``cam_name`` entries are
          ``("frame_t_rgb", "frame_tk_rgb")`` — one slot per pair-frame, in
          that order.

    The class layout mirrors ``ValueCriticModel`` so FSDP wrap names
    and the offline pipeline keep working unchanged.
    """

    def __init__(self, config: SteamConfig):
        super().__init__()
        self.config = config
        self.model = SteamBackbone(config)
        self.label_smoothing = float(config.label_smoothing)
        self.gradient_checkpointing_enabled = False

        # FSDP wrap-name tagging (mirrors recap/modeling_critic.py:160-162)
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "SiglipVisionEmbeddings",
            "Gemma3RotaryEmbedding",
            "LayerNorm",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "image_projector",
            "language_projector",
            "value_head",
        ]

    def gradient_checkpointing_enable(self):
        if self.gradient_checkpointing_enabled:
            return
        self.gradient_checkpointing_enabled = True
        for submod in (self.model.vision_encoder, self.model.language_model):
            if hasattr(submod, "gradient_checkpointing_enable"):
                submod.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
        logger.info("Enabled gradient checkpointing for SteamCriticModel")

    def gradient_checkpointing_disable(self):
        if not self.gradient_checkpointing_enabled:
            return
        self.gradient_checkpointing_enabled = False
        for submod in (self.model.vision_encoder, self.model.language_model):
            if hasattr(submod, "gradient_checkpointing_disable"):
                submod.gradient_checkpointing_disable()
        logger.info("Disabled gradient checkpointing for SteamCriticModel")

    def attach_runtime_assets(self, processor, device) -> None:
        """Attach inference-time runtime assets to this model instance.

        Called from :meth:`from_checkpoint` to wire up the processor and
        device target. STEAM is a pair model whose inference contract is
        collator-prepared observations, so raw-observation OpenPI transforms
        are intentionally not attached here.
        """
        self.processor = processor
        self._device = device

    # ------------------------------------------------------------------
    # Observation -> tensor adapter
    # ------------------------------------------------------------------

    def _stack_observation(
        self, observation: dict
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Convert collator observation dict to :class:`SteamBackbone` args.

        Expects each ``observation["images"][cam_name]`` to be shape
        ``[B, num_frames, 3, H, W]`` — the camera axis is the dict key,
        and the frame axis is stacked within the tensor by
        :class:`BinaryPairDataCollator`.

        Returns:
            ``(input_ids[B,T], attention_mask[B,T],
              images[B, num_cameras, num_frames, 3, H, W],
              image_attention_mask[B, num_cameras, num_frames])``.
        """
        images_dict = observation["images"]
        image_masks_dict = observation.get("image_masks", {})
        input_ids = observation["tokenized_prompt"]
        attention_mask = observation["tokenized_prompt_mask"]

        sorted_cams = sorted(images_dict.keys())
        img_list = [images_dict[k] for k in sorted_cams]
        if not img_list:
            raise ValueError("observation['images'] is empty")
        template = img_list[0]
        if template.ndim != 5:
            raise ValueError(
                "observation['images'][cam] must have shape "
                f"[B, num_frames, C, H, W]; got {tuple(template.shape)}"
            )
        bsize, num_frames = template.shape[0], template.shape[1]
        device = template.device

        # Camera axis is the dict key — stack along a new dim=1.
        images_stacked = torch.stack(img_list, dim=1)
        # images_stacked: [B, num_cameras, num_frames, C, H, W]

        mask_list = [
            image_masks_dict.get(
                k,
                torch.ones(bsize, num_frames, dtype=torch.bool, device=device),
            )
            for k in sorted_cams
        ]
        # Masks can be either [B, num_frames] (per-camera) or [B] (treated
        # as a per-camera scalar broadcast across frames).
        normalised_masks: list[Tensor] = []
        for m in mask_list:
            if m.ndim == 1:
                m = m.unsqueeze(-1).expand(-1, num_frames)
            normalised_masks.append(m.to(torch.bool))
        image_mask_stacked = torch.stack(normalised_masks, dim=1)
        # image_mask_stacked: [B, num_cameras, num_frames]
        return (
            input_ids.long(),
            attention_mask.long(),
            images_stacked,
            image_mask_stacked,
        )

    # ------------------------------------------------------------------
    # Loss — ``num_bins``-way cross-entropy (covers binary and multi-bin)
    # ------------------------------------------------------------------

    def _compute_loss(self, logits, bin_labels):
        """``num_bins``-way cross-entropy on bin indices.

        Paired with :class:`~rlinf.data.datasets.steam.PairDataset`,
        which emits long bin indices in ``[0, num_bins)`` for both the
        binary (``num_bins == 2``) and multi-bin (``num_bins > 2``)
        modes. The signed-stride → bin mapping is owned by the dataset's
        ``_signed_stride_to_bin`` helper.

        Args:
            logits: Shape ``[B, num_bins]``.
            bin_labels: Shape ``[B]`` with values in ``[0, num_bins)``.

        Returns:
            Tuple of (per-sample loss, metrics dict). Metrics:
                * ``acc_best`` — exact-bin classification accuracy.
                * ``acc_neighbor`` — ``|pred_bin - target_bin| ≤ 1``.
                  Trivially 1 when ``num_bins == 2`` (no non-neighbor
                  bins exist); kept for ``CriticOutput`` slot parity.
                * ``mae`` — zero (no scalar target in this mode; kept
                  for ``CriticOutput`` slot parity with the categorical
                  variant).
        """
        num_bins = int(self.config.num_bins)
        if logits.ndim != 2 or logits.shape[-1] != num_bins:
            raise ValueError(
                f"logits must have shape [B, {num_bins}], got {tuple(logits.shape)}"
            )
        targets = bin_labels.to(dtype=torch.long)
        if targets.ndim != 1:
            raise ValueError(
                f"bin_labels must be rank-1, got {tuple(bin_labels.shape)}"
            )
        if int(targets.min().item()) < 0 or int(targets.max().item()) >= num_bins:
            raise ValueError(
                f"bin_labels out of range [0, {num_bins}); "
                f"min={int(targets.min().item())}, max={int(targets.max().item())}"
            )
        loss = F.cross_entropy(
            logits,
            targets,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )

        pred_class = logits.argmax(dim=-1)
        acc_best = (pred_class == targets).to(dtype=torch.float32).mean()
        acc_neighbor = (
            ((pred_class - targets).abs() <= 1).to(dtype=torch.float32).mean()
        )

        metrics = {
            "acc_best": acc_best,
            "acc_neighbor": acc_neighbor,
            "mae": torch.zeros((), device=logits.device),
        }
        return loss, metrics

    def _predicted_signed_value(self, probs: Tensor) -> Tensor:
        """Return a signed bin-weighted expectation in ``[-1, 1]`` (shape ``[B]``).

        Computes ``E[signed_bin] / half`` where ``half = num_bins // 2``
        and each bin gets an integer signed position:

            * bins ``[0, half)``            → signed values ``[-half, -1]``
              (regressive, matching the negative-stride half of
              ``pair_dataset._signed_stride_to_bin``).
            * bins ``[half, num_bins)``     → signed values ``[1, half]``
              (progressive).

        Dividing by ``half`` maps the raw expectation (range
        ``[-half, half]``) onto ``[-1, 1]``. Using the full distribution
        means a bin at the extreme (strong progress / strong regress)
        contributes with larger magnitude than a near-midpoint bin, so the
        score carries both direction and strength.

        Binary (``num_bins == 2``) degenerates to
        ``-p[:, 0] + p[:, 1] = 2 · P(progress) - 1``, matching the
        signed-confidence convention documented on
        :func:`~rlinf.data.datasets.steam.binning.bin_centers`.
        """
        num_bins = int(self.config.num_bins)
        half = num_bins // 2
        if half < 1:
            raise ValueError(
                f"_predicted_signed_value requires num_bins >= 2 and even; "
                f"got num_bins={num_bins}."
            )
        arange = torch.arange(num_bins, device=probs.device, dtype=probs.dtype)
        signed_bin = torch.where(
            arange < half,
            arange - float(half),  # [0, half) -> [-half, -1]
            arange - float(half) + 1.0,  # [half, num_bins) -> [1, half]
        )
        return (probs * signed_bin).sum(dim=-1) / float(half)

    # ------------------------------------------------------------------
    # Forward / predict
    # ------------------------------------------------------------------

    def forward(self, observation, labels=None, **kwargs) -> CriticOutput:
        """Forward pass — parallel to ValueCriticModel.forward.

        Stacks the observation, runs the multimodal backbone, takes
        softmax over the ``num_bins``-wide head, and — if ``labels`` are
        provided — computes cross-entropy + accuracy via
        :meth:`_compute_loss`. Returns a fully populated
        :class:`CriticOutput`.
        """
        input_ids, attention_mask, images, image_mask = self._stack_observation(
            observation
        )
        # SteamImageProcessor emits [0, 1] BCHW images; the
        # backbone applies SigLIP-style mean/std normalization internally.
        hidden_states, _per_frame, _language = self.model._compute_projected_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_mask,
        )  # [B, fusion_hidden_dim * (num_frames_per_pair + 1)]
        value_head_dtype = _module_parameter_dtype(
            self.model.value_head,
            hidden_states.dtype,
        )
        logits = self.model.value_head(
            hidden_states.to(dtype=value_head_dtype)
        )  # [B, num_bins]

        probs = F.softmax(logits, dim=-1)  # [B, num_bins]
        progress_values = self._predicted_signed_value(probs)  # [B]
        predicted_values = progress_values

        expert_loss = None
        cat_metrics = None
        if labels is not None:
            expert_loss, cat_metrics = self._compute_loss(logits, labels)

        expert_loss_mean = expert_loss.mean() if expert_loss is not None else None
        total_loss = expert_loss_mean

        return CriticOutput(
            loss=total_loss,
            predicted_values=predicted_values,
            logits=logits,
            probs=probs,
            atoms=None,
            expert_loss=expert_loss_mean,
            hidden_states=hidden_states,
            cat_acc_best=cat_metrics["acc_best"] if cat_metrics else None,
            cat_acc_neighbor=cat_metrics["acc_neighbor"] if cat_metrics else None,
            mae=cat_metrics["mae"] if cat_metrics else None,
            progress_values=progress_values,
        )

    @torch.no_grad()
    def predict(self, observation) -> CriticOutput:
        """Inference forward — parallel to ValueCriticModel.predict.

        Separate from :meth:`forward` so the inference path stays cheap and
        mirrors ``ValueCriticModel``'s structure. The returned ``CriticOutput``
        populates only inference-relevant fields; loss and metric fields
        stay at their dataclass defaults of ``None``.
        """
        input_ids, attention_mask, images, image_mask = self._stack_observation(
            observation
        )
        hidden_states, _per_frame, _language = self.model._compute_projected_features(
            input_ids=input_ids,
            attention_mask=attention_mask,
            images=images,
            image_attention_mask=image_mask,
        )
        value_head_dtype = _module_parameter_dtype(
            self.model.value_head,
            hidden_states.dtype,
        )
        logits = self.model.value_head(
            hidden_states.to(dtype=value_head_dtype)
        )  # [B, num_bins]
        probs = F.softmax(logits, dim=-1)  # [B, num_bins]

        progress_values = self._predicted_signed_value(probs)
        predicted_values = progress_values
        return CriticOutput(
            predicted_values=predicted_values,
            logits=logits,
            probs=probs,
            atoms=None,
            hidden_states=hidden_states,
            progress_values=progress_values,
        )

    @torch.no_grad()
    def predict_value(self, observation) -> Tensor:
        """Return the signed value in ``[-1, 1]`` per sample (shape ``[B]``).

        This is ``predicted_values`` — for the binary head it equals
        ``2 · P(progress) - 1``, not ``P(progress)`` itself.
        """
        return self.predict(observation).predicted_values

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_dir,
        *,
        device: str = "cuda",
        env_type: str = "libero",
        model_type: str = "pi05",
        default_prompt: Optional[str] = None,
        norm_stats: Optional[dict] = None,
        label_smoothing: Optional[float] = None,
        num_frames_per_pair: Optional[int] = None,
        num_bins: Optional[int] = None,
        stride_k: Optional[int] = None,
        ensemble_size: Optional[int] = None,
        precision: Optional[str] = None,
        ensemble_head_seed_base: Optional[int] = None,
        tokenizer_path: Optional[str] = None,
        vision_repo_id: Optional[str] = None,
        language_repo_id: Optional[str] = None,
        fusion_hidden_dim: Optional[int] = None,
        dropout: Optional[float] = None,
        # Prompt tokenization length — must match training config.
        max_token_len: Optional[int] = None,
        **kwargs,
    ) -> "SteamCriticModel | EnsembleSteamCriticModel":
        """Build a STEAM value critic from a checkpoint, ready for inference.

        Dispatches through :func:`get_model`: returns a single-model
        :class:`SteamCriticModel` when ``ensemble_size == 1``, or an
        :class:`~rlinf.models.embodiment.value_model.steam.\
ensemble_modeling_critic.EnsembleSteamCriticModel` wrapper when
        ``ensemble_size > 1``. Either return value exposes the same
        ``predict`` surface; callers that want ensemble aggregate stats
        must call the ensemble-specific paths.
        Mirrors ``ValueCriticModel.from_checkpoint`` so the offline
        pipeline (compute_advantages, etc.) dispatches on ``model_type`` and
        loads either variant via the same call shape.
        """
        import pathlib

        from omegaconf import OmegaConf
        from transformers import AutoTokenizer

        from . import get_model
        from .processing import SteamImageProcessor, SteamProcessor

        del env_type, model_type, default_prompt, norm_stats, kwargs

        checkpoint_dir = pathlib.Path(checkpoint_dir)
        logger.info(f"Loading STEAM value model from {checkpoint_dir}")

        cfg_dict = {"model_path": str(checkpoint_dir)}
        optional_overrides = {
            "vision_repo_id": vision_repo_id,
            "language_repo_id": language_repo_id,
            "label_smoothing": label_smoothing,
            "num_frames_per_pair": num_frames_per_pair,
            "num_bins": num_bins,
            "stride_k": stride_k,
            "ensemble_size": ensemble_size,
            "precision": precision,
            "ensemble_head_seed_base": ensemble_head_seed_base,
            "fusion_hidden_dim": fusion_hidden_dim,
            "dropout": dropout,
            "max_token_len": max_token_len,
        }
        for key, value in optional_overrides.items():
            if value is not None:
                cfg_dict[key] = value

        cfg = OmegaConf.create(cfg_dict)
        model = get_model(cfg)

        # Tokenizer resolution
        tokenizer_source, tokenizer_local_only = _resolve_tokenizer_source(
            checkpoint_dir=checkpoint_dir,
            explicit_tokenizer_path=tokenizer_path,
            model_config=model.config,
        )
        if tokenizer_path:
            logger.info("  Using explicit tokenizer_path: %s", tokenizer_source)
        elif str(tokenizer_source) == str(checkpoint_dir):
            logger.info("  Found tokenizer files in checkpoint")
        else:
            logger.info(
                "  Falling back to tokenizer source from model config: %s",
                tokenizer_source,
            )

        tokenizer_kwargs = {"add_bos_token": True}
        if tokenizer_local_only:
            tokenizer_kwargs["local_files_only"] = True
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            **tokenizer_kwargs,
        )

        image_processor = None
        try:
            image_processor = SteamImageProcessor.from_pretrained(
                str(checkpoint_dir),
                local_files_only=True,
            )
            logger.info("  Found image processor config in checkpoint")
        except (OSError, ValueError):
            logger.info(
                "  No image processor config found in checkpoint; using defaults"
            )

        member = model.members[0] if hasattr(model, "members") else model
        backbone = getattr(member, "model")
        model_image_size = tuple(int(v) for v in backbone.image_resolution)
        if image_processor is None:
            image_processor = SteamImageProcessor(image_size=model_image_size)
        elif tuple(image_processor.image_size) != model_image_size:
            logger.warning(
                "  Checkpoint image processor size %s does not match vision "
                "encoder native size %s; preserving the checkpoint processor "
                "for preprocessing parity",
                tuple(image_processor.image_size),
                model_image_size,
            )

        processor = SteamProcessor(
            image_processor=image_processor,
            tokenizer=tokenizer,
            max_token_len=getattr(model.config, "max_token_len", 200),
        )

        model.attach_runtime_assets(
            processor=processor,
            device=device,
        )

        model = model.to(device)
        model.eval()

        logger.info("SteamCriticModel.from_checkpoint ready for inference")
        return model


__all__ = [
    "SteamCriticModel",
    "CriticOutput",
]
