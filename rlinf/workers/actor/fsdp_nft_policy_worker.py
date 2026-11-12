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

import numpy as np
import torch

from rlinf.algorithms.diffusion.nft import (
    build_nft_schedule,
    compute_nft_loss,
    get_nft_tau,
    nft_requires_ema,
    resample_nft_inputs,
    update_nft_ema,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler.worker.worker import Worker
from rlinf.utils.distributed import all_reduce_dict
from rlinf.utils.metric_utils import append_to_dict
from rlinf.utils.nested_dict_process import (
    process_nested_dict_for_train,
    put_tensor_device,
    split_dict_to_chunk,
)
from rlinf.utils.pinned_offload import pinned_offload_context
from rlinf.utils.utils import clear_memory
from rlinf.workers.actor.embodied_fsdp_actor_worker import EmbodiedFSDPActor


class EmbodiedNFTFSDPPolicy(EmbodiedFSDPActor):
    """Embodied FSDP policy worker for NFT with off-policy support."""

    # =======================================================================
    # Initialization & Rollout Model Management
    # =======================================================================

    def init_worker(self) -> None:
        """Initialize actor and rollout model state for off-policy support."""
        super().init_worker()
        self.init_rollout_model()

    def offload_param_and_grad(self, offload_grad: bool = False) -> None:
        """Override to also offload ``rollout_model_state_dict`` to CPU.

        The parent implementation only moves FSDP model parameters and
        gradients to CPU.  However, ``rollout_model_state_dict`` is a full
        (un-sharded) copy of the model weights that stays on GPU if not
        explicitly moved — consuming sizeof(bf16) x number of params of GPU memory
        even during ``recv_rollout_trajectories`` when the model itself is
        offloaded.  By piggybacking on the parent's offload, the rollout
        state dict follows the same lifecycle: on CPU when the model is
        offloaded (recv phase), on GPU when the model is loaded (training
        phase).  Pinned memory is used for the GPU→CPU copy to maximise PCIe
        bandwidth.
        """
        super().offload_param_and_grad(offload_grad)
        rollout_state = getattr(self, "rollout_model_state_dict", None)
        if rollout_state:
            with pinned_offload_context():
                for key, value in rollout_state.items():
                    if torch.is_tensor(value) and value.device.type != "cpu":
                        rollout_state[key] = value.to("cpu")

    def load_param_and_grad(self, device_id: int, load_grad: bool = False) -> None:
        """Override to also load ``rollout_model_state_dict`` back to GPU.

        Mirrors :meth:`offload_param_and_grad`: when the FSDP parameters are
        loaded to GPU for training, also move ``rollout_model_state_dict``
        so that ``soft_update_rollout_model`` (EMA update via ``lerp_``) and
        ``_recompute_v_old`` (``load_state_dict`` into the ref model) can
        operate on-GPU without cross-device copies.
        """
        super().load_param_and_grad(device_id, load_grad)
        rollout_state = getattr(self, "rollout_model_state_dict", None)
        if rollout_state:
            device = (
                torch.device(device_id)
                if not isinstance(device_id, torch.device)
                else device_id
            )
            for key, value in rollout_state.items():
                if torch.is_tensor(value) and value.device.type == "cpu":
                    rollout_state[key] = value.to(device)

    def init_rollout_model(self) -> None:
        """Initialize rollout model state for off-policy support."""
        self.rollout_model_state_dict = {}
        self._cached_ref_model = None  # lazy-init, reused across iterations
        if nft_requires_ema(self.cfg.algorithm.get("nft_tau", 1.0)):
            src = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
            for key, value in src.items():
                if torch.is_tensor(value):
                    self.rollout_model_state_dict[key] = value.clone()
                else:
                    self.rollout_model_state_dict[key] = value

    def save_checkpoint(self, save_path: str, step: int = 0) -> None:
        """Save actor training state and the replicated NFT EMA weights."""
        super().save_checkpoint(save_path, step)
        if not self.rollout_model_state_dict:
            return

        path = os.path.join(save_path, "nft_rollout_state.pt")
        status = [None]
        save_error = None
        if self._rank == 0:
            try:
                state = {
                    key: value.detach().cpu() if torch.is_tensor(value) else value
                    for key, value in self.rollout_model_state_dict.items()
                }
                torch.save(state, path + ".tmp")
                os.replace(path + ".tmp", path)
            except Exception as error:
                save_error = error
                status[0] = f"Failed to save NFT EMA weights to '{path}': {error}"
        torch.distributed.broadcast_object_list(status, src=0)
        if status[0] is not None:
            raise RuntimeError(status[0]) from save_error

    def load_checkpoint(self, load_path: str) -> None:
        """Restore actor training state and NFT EMA weights, if present."""
        super().load_checkpoint(load_path)
        if not self.rollout_model_state_dict:
            return

        path = os.path.join(load_path, "nft_rollout_state.pt")
        if not os.path.isfile(path):
            self.log_warning(
                f"NFT checkpoint '{path}' has no EMA weights; initializing EMA "
                "from the restored actor. This is not an exact training resume."
            )
            self.init_rollout_model()
            return

        state = torch.load(path, map_location="cpu", weights_only=True)
        expected_keys = set(self.rollout_model_state_dict)
        if set(state) != expected_keys:
            raise ValueError(
                "NFT EMA checkpoint keys do not match the actor: "
                f"missing={sorted(expected_keys - set(state))}, "
                f"unexpected={sorted(set(state) - expected_keys)}."
            )
        for key, target in self.rollout_model_state_dict.items():
            value = state[key]
            if torch.is_tensor(target) and (
                not torch.is_tensor(value)
                or value.shape != target.shape
                or value.dtype != target.dtype
            ):
                raise ValueError(
                    f"NFT EMA checkpoint tensor '{key}' expected shape "
                    f"{tuple(target.shape)} and dtype {target.dtype}, found "
                    f"shape {getattr(value, 'shape', None)} and dtype "
                    f"{getattr(value, 'dtype', None)}."
                )
        with torch.no_grad():
            for key, target in self.rollout_model_state_dict.items():
                if torch.is_tensor(target):
                    target.copy_(state[key])
                else:
                    self.rollout_model_state_dict[key] = state[key]
        self._cached_ref_model = None

    def _get_current_nft_tau(self) -> float:
        """Return the rollout EMA update rate for this actor version."""
        return get_nft_tau(
            self.cfg.algorithm.get("nft_tau", 1.0), int(getattr(self, "version", 0))
        )

    def get_rollout_state_dict(self) -> dict:
        """Return EMA cache when tau<1, otherwise defer to the on-policy parent behavior."""
        if self._get_current_nft_tau() < 1.0:
            return self.rollout_model_state_dict
        return super().get_rollout_state_dict()

    def soft_update_rollout_model(self) -> None:
        """Update the replicated EMA cache from the current full actor state."""
        if not self.rollout_model_state_dict:
            return
        tau = self._get_current_nft_tau()
        current = self.get_model_state_dict(cpu_offload=False, full_state_dict=True)
        update_nft_ema(self.rollout_model_state_dict, current, tau)

    # =======================================================================
    # Training Loop & Data Preprocessing
    # =======================================================================

    @Worker.timer("run_training")
    def run_training(self) -> dict:
        """Run NFT training with off-policy decay support."""
        tau = self._get_current_nft_tau()
        if tau >= 1.0:
            # On-policy: load model immediately since precompute doesn't
            # need a ref model and the model is used throughout.
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

        self.model.train()
        rollout_size = (
            self.rollout_batch["prev_logprobs"].shape[0]
            * self.rollout_batch["prev_logprobs"].shape[1]
        )
        g = torch.Generator()
        g.manual_seed(self.cfg.actor.seed + self._rank)
        shuffle_id = torch.randperm(rollout_size, generator=g)

        with torch.no_grad():
            self.rollout_batch = process_nested_dict_for_train(
                self.rollout_batch, shuffle_id
            )
            self._precompute_nft_training_inputs()

        if tau < 1.0:
            # Off-policy: defer loading the training model until *after*
            # precompute.  During precompute (_recompute_v_old), the training
            # model is offloaded to make room for the ref model, so loading
            # it at the top would be immediately undone — a wasted GPU→CPU→GPU
            # round trip.  By deferring, the model stays on CPU through the
            # entire precompute phase and is only loaded when training begins.
            if self.is_weight_offloaded:
                self.load_param_and_grad(self.device)
            if self.is_optimizer_offloaded:
                self.load_optimizer(self.device)

        assert (
            self.cfg.actor.global_batch_size
            % (self.cfg.actor.micro_batch_size * self._world_size)
            == 0
        ), "global_batch_size is not divisible by micro_batch_size * world_size"

        self.gradient_accumulation = (
            self.cfg.actor.global_batch_size
            // self.cfg.actor.micro_batch_size
            // self._world_size
        )

        rollout_size = self.rollout_batch["prev_logprobs"].size(0)
        batch_size_per_rank = self.cfg.actor.global_batch_size // self._world_size
        assert rollout_size % batch_size_per_rank == 0, (
            f"{rollout_size} is not divisible by {batch_size_per_rank}"
        )
        metrics = {}
        update_epoch = self.cfg.algorithm.get("update_epoch", 1)
        for _ in range(update_epoch):
            rollout_dataloader_iter = split_dict_to_chunk(
                self.rollout_batch,
                rollout_size // batch_size_per_rank,
            )
            for train_global_batch in rollout_dataloader_iter:
                train_global_batch_size = train_global_batch["prev_logprobs"].shape[0]
                assert (
                    train_global_batch_size
                    == self.cfg.actor.global_batch_size
                    // torch.distributed.get_world_size()
                )
                assert train_global_batch_size % self.cfg.actor.micro_batch_size == 0, (
                    f"{train_global_batch_size=}, {self.cfg.actor.micro_batch_size}"
                )

                train_micro_batch = split_dict_to_chunk(
                    train_global_batch,
                    train_global_batch_size // self.cfg.actor.micro_batch_size,
                )

                self.optimizer.zero_grad()
                for idx, batch in enumerate(train_micro_batch):
                    batch = put_tensor_device(
                        batch,
                        f"{Worker.torch_device_type}:{int(os.environ['LOCAL_RANK'])}",
                    )
                    backward_ctx = self.before_micro_batch(
                        self.model,
                        is_last_micro_batch=(idx + 1) == self.gradient_accumulation,
                    )

                    loss, metrics_data = self.nft_forward_and_loss(batch)

                    if self.enable_sft_co_train:
                        loss = self._train_sft_epoch(metrics_data, loss)

                    loss /= self.gradient_accumulation
                    with backward_ctx:
                        self.grad_scaler.scale(loss).backward()

                    metrics_data["actor/total_loss"] = loss.detach().item()
                    append_to_dict(metrics, metrics_data)
                    # avoid gpu memory leak
                    train_micro_batch[idx] = None
                    del batch, loss, metrics_data

                self.torch_platform.empty_cache()

                grad_norm, lr_list = self.optimizer_step()
                data = {
                    "actor/grad_norm": grad_norm,
                    "actor/lr": lr_list[0],
                }
                if len(lr_list) > 1:
                    data["critic/lr"] = lr_list[1]
                append_to_dict(metrics, data)
        # put LR scheduler step here
        self.lr_scheduler.step()
        self.soft_update_rollout_model()
        self.optimizer.zero_grad()
        clear_memory()
        mean_metric_dict = {key: np.mean(value) for key, value in metrics.items()}
        mean_metric_dict = all_reduce_dict(
            mean_metric_dict, op=torch.distributed.ReduceOp.AVG
        )
        return mean_metric_dict

    def _precompute_nft_training_inputs(self) -> None:
        """Prepare NFT training tensors before the update loop."""
        forward_inputs = self.rollout_batch["forward_inputs"]
        xcur_source = self.cfg.algorithm.get("nft_xcur_source", "rollout")
        num_steps = self.model.config.num_steps
        recompute_v = bool(self.cfg.algorithm.get("recompute_v", False))

        if xcur_source == "resample":
            # Resample step indices and interpolate xcur from x0
            x0 = forward_inputs["nft_x0"]
            xcur, step_indices = resample_nft_inputs(x0, num_steps)
            forward_inputs["nft_xcur"] = xcur
            forward_inputs["nft_step_index"] = step_indices
            recompute_v = True  # must recompute v_old for resampled xcur

        if recompute_v:
            # recompute v_old: always for resample, opt-in for rollout
            xcur = forward_inputs["nft_xcur"]
            step_indices = forward_inputs["nft_step_index"]
            _, t = build_nft_schedule(
                step_indices, xcur.device, xcur.dtype, num_steps=num_steps
            )
            forward_inputs["nft_v"] = self._recompute_v_old(forward_inputs, xcur, t)

    def _recompute_v_old(
        self,
        forward_inputs: dict,
        xcur: torch.Tensor,
        t: torch.Tensor,
    ) -> torch.Tensor:
        """Recompute the old velocity with the rollout model."""
        micro_bs = self.cfg.actor.micro_batch_size
        v_old_buffer = []
        training_was_on_device = False
        cleanup_rollout_model = False

        tau = self._get_current_nft_tau()
        if tau >= 1.0:
            # On-policy: use training model directly
            ref_model = self.model
        else:
            # Off-policy: use a cached ref model with lagged rollout weights.
            # The model is built once and reused across iterations; only
            # load_state_dict + to(device) is needed each time.
            training_was_on_device = not self.is_weight_offloaded
            if training_was_on_device:
                self.offload_param_and_grad()
                clear_memory()

            if self._cached_ref_model is None:
                ref_model = self.model_provider_func()
                ref_config = getattr(ref_model, "config", None)
                # Compiling the reference model adds repeated compilation and cache overhead.
                if hasattr(ref_config, "compile_transformer_forward"):
                    ref_config.compile_transformer_forward = False
                ref_model.eval()
                ref_model.requires_grad_(False)
                prepare_reference = getattr(
                    ref_model, "prepare_nft_reference_model", None
                )
                if callable(prepare_reference):
                    prepare_reference()
                self._cached_ref_model = ref_model
            else:
                ref_model = self._cached_ref_model

            ref_model.load_state_dict(self.get_rollout_state_dict(), strict=False)
            ref_model.to(self.device)
            cleanup_rollout_model = True

        with torch.no_grad():
            for start in range(0, xcur.shape[0], micro_bs):
                end = min(start + micro_bs, xcur.shape[0])
                fi_slice = put_tensor_device(
                    self._slice_forward_inputs(forward_inputs, start, end),
                    self.device,
                )
                out = ref_model(
                    forward_type=ForwardType.NFT,
                    forward_inputs=fi_slice,
                    nft_inputs={
                        "x_t": xcur[start:end].to(device=self.device),
                        "timesteps": t[start:end].to(device=self.device),
                    },
                    compute_values=False,
                )
                v_old_buffer.append(out["v_theta"].detach().cpu())

        if cleanup_rollout_model:
            with pinned_offload_context():
                self._cached_ref_model.to("cpu")
            clear_memory()
            if training_was_on_device:
                self.load_param_and_grad(self.device)

        return torch.cat(v_old_buffer, dim=0).to(xcur.device)

    # =======================================================================
    # NFT Forward & Loss
    # =======================================================================

    def nft_forward_and_loss(
        self, batch: dict
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the actor forward pass and evaluate the NFT tensor objective."""
        forward_inputs = batch["forward_inputs"]
        x_t = forward_inputs["nft_xcur"]
        schedule, timesteps = build_nft_schedule(
            forward_inputs["nft_step_index"],
            x_t.device,
            x_t.dtype,
            num_steps=self.model.config.num_steps,
        )
        with self.amp_context:
            output = self.model(
                forward_type=ForwardType.NFT,
                forward_inputs=forward_inputs,
                nft_inputs={"x_t": x_t, "timesteps": timesteps},
                compute_values=False,
            )
        loss, metrics = compute_nft_loss(
            output["v_theta"],
            forward_inputs,
            batch["advantages"],
            batch["loss_mask"],
            schedule=schedule,
            config=self.cfg.algorithm,
        )
        metrics["actor/nft_tau"] = self._get_current_nft_tau()
        return loss, metrics

    def _slice_forward_inputs(self, forward_inputs: dict, start: int, end: int) -> dict:
        """Slice nested forward inputs along the batch dimension."""
        ret = {}
        for key, value in forward_inputs.items():
            if isinstance(value, torch.Tensor):
                ret[key] = value[start:end]
            elif isinstance(value, dict):
                ret[key] = self._slice_forward_inputs(value, start, end)
            else:
                ret[key] = value
        return ret
