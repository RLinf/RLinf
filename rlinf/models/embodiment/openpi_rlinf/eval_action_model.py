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

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal, Sequence

import numpy as np
import os
import torch
from torch.utils._pytree import tree_map

from rlinf.models.embodiment.openpi_rlinf.openpi_action_model import (
    OpenPiPytorchActionModel,
)
from rlinf.models.embodiment.openpi_rlinf.pi0_model import model as pi0_model_module
from rlinf.models.embodiment.openpi_rlinf.pi0_model.model import Observation
from rlinf.models.embodiment.openpi_rlinf.pi0_model.pi0 import Pi0
from rlinf.models.embodiment.openpi_rlinf.utils.rlt_utils import (
    OpenPiPytorchRLTConfig,
)


def _to_numpy(x):
    return np.asarray(x.detach().cpu()) if torch.is_tensor(x) else x


# Tianji Stage2 RTC: rows of the in-flight motion's constant-velocity
# continuation that also guide the reference chunk (the first executed row
# hard, the rest soft). The actor chunk starts right after the in-flight rows,
# where the previous plan ends; with those rows alone the first executed row
# was free and missed the continuation by 13.5 mm at p90 (47 mm max) on
# recorded frames. 20 rows: no junction error, at most 3.7 mm per step (p90)
# into the new plan, 0.16 mm change to its last 24 rows. Stage1 gets the same
# from the rest of its previous plan.
TIANJI_RTC_CONTINUATION_ROWS = 20
# Positions and rot6 of both arms; the commanded closures are held.
_TIANJI_POSE_DIMS = [d for d in range(20) if d not in (9, 19)]
# rtj fork: the 16D joint Stage1 checkpoint (state = image-aligned measured
# joints + commanded closures; output = absolute joints + closures).
TIANJI_JOINT_CONFIG = "tianji_stage1_joint"
_TIANJI_JOINT_DIMS = [d for a0 in (0, 8) for d in range(a0, a0 + 7)]


class OpenPiPytorchEvalActionModel(OpenPiPytorchActionModel):
    """Eval-capable wrapper around the vendored ``Pi0`` model.

    Drives observation construction through the upstream ``openpi.transforms``
    pipeline: the factory calls :meth:`setup_wrappers` with the composed
    input/output transform lists from :func:`get_openpi_config`, then
    :meth:`predict_action_batch` routes ``env_obs`` through
    :meth:`_repack_env_obs` → :meth:`input_transform` →
    :meth:`_observation_dict_to_device` → :meth:`Pi0.sample_actions` →
    :meth:`output_transform`.

    The RL subclass inherits this eval path unchanged and adds the PPO
    chain-collecting SDE sampler on top.
    """

    def __init__(
        self,
        pi0_model: Pi0,
        *,
        num_steps: int,
        action_env_dim: int,
        action_chunk: int | None = None,
        config_name: str = "",
        state_indices: Sequence[int] | None = None,
        rlt_cfg: OpenPiPytorchRLTConfig | None = None,
    ):
        super().__init__(
            pi0_model,
            num_steps=num_steps,
            action_env_dim=action_env_dim,
            rlt_cfg=rlt_cfg,
        )
        # ``action_chunk`` slices the env-action subspace from the model output
        # in :meth:`output_transform`.
        self.action_chunk = action_chunk
        # ``config_name`` is the openpi TrainConfig key (e.g. ``pi05_behavior``)
        # used by :meth:`_repack_env_obs` to switch the env→``observation/*``
        # repack rules (currently only the ``calvin`` split-state layout).
        self.config_name = config_name
        # Optional subset of the raw env state dim (openpi ``state_indices``).
        # ``None`` (the BEHAVIOR default) is an identity passthrough.
        self.state_indices = list(state_indices) if state_indices else None

        # openpi.transforms pipeline state (installed by :meth:`setup_wrappers`).
        self._input_transform_fn = None
        self._output_transform_fn = None

    # -------------------------------------------------------- transforms glue

    def setup_wrappers(
        self,
        transforms: Sequence,
        output_transforms: Sequence,
    ) -> None:
        """Install the openpi.transforms input/output pipelines.

        ``transforms`` is the list passed to the openpi-side ``compose`` —
        typically ``BehaviorInputs → Normalize(norm_stats) → ModelTransformFactory(model_config)``
        (the last stage carries the auto-downloading PaliGemma tokenizer).
        ``output_transforms`` is the matching reverse pipeline used to turn
        sampled model actions back into env-frame actions.
        """
        from openpi.transforms import compose

        self._input_transform_fn = compose(transforms)
        self._output_transform_fn = compose(output_transforms)

    def _ensure_wrappers(self) -> None:
        if self._input_transform_fn is None or self._output_transform_fn is None:
            raise RuntimeError(
                f"{type(self).__name__}.setup_wrappers(...) must be called "
                "after construction (the factory does this); the openpi "
                "transforms pipeline is not yet installed."
            )

    def _select_configured_state(self, states):
        """Select a configured subset of the raw env state (openpi parity).

        Mirrors ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.
        _select_configured_state``: ``state_indices=None`` (BEHAVIOR) is an
        identity passthrough; a configured index list fancy-indexes the last
        dim so future non-BEHAVIOR envs can reuse the generic repack.
        """
        indices = self.state_indices
        if not indices:
            return states

        if hasattr(states, "shape"):
            state_dim = states.shape[-1]
        else:
            state_dim = np.asarray(states).shape[-1]
        if state_dim == len(indices):
            return states
        if state_dim <= max(indices):
            raise ValueError(
                f"Cannot select state_indices={indices} from state dim {state_dim}."
            )

        if torch.is_tensor(states):
            index_tensor = torch.as_tensor(indices, device=states.device)
            return states.index_select(-1, index_tensor)
        return np.asarray(states)[..., indices]

    def _repack_env_obs(self, env_obs: dict) -> dict:
        """Map the env's observation dict to the ``observation/*`` keys the
        openpi pipeline expects.

        Generic adapter mirroring
        ``openpi.openpi_action_model.OpenPi0ForRLActionPrediction.obs_processor``:
        it reads the standardized RLinf env keys (``states`` / ``main_images`` /
        ``task_descriptions`` and the optional ``wrist_images`` /
        ``extra_view_images``) and repacks them into openpi's ``observation/*``
        convention. All real per-env logic lives in the config-driven
        ``data_transforms.inputs`` (selected by ``config_name``), not here; the
        only per-env branch is the ``calvin`` split-state layout. Optional
        camera views are added only when the env actually provides them
        (BEHAVIOR, for instance, emits no ``extra_view_images`` key).
        """
        env_states = self._select_configured_state(
            env_obs["joint_states"] if self.config_name == TIANJI_JOINT_CONFIG else env_obs["states"]
        )
        if "aloha" in self.config_name:
            # AlohaInputs 约定:images 为 dict[name, CHW float32 0..1]
            def _chw(img):
                if torch.is_tensor(img):
                    img = img.cpu().numpy()
                x = np.asarray(img, dtype=np.float32)
                if x.max() > 1.5:
                    x = x / 255.0
                return np.moveaxis(x, -1, -3)

            images = {"cam_high": _chw(env_obs["main_images"])}
            extra = env_obs.get("extra_view_images")
            if extra is not None:
                images["cam_left_wrist"] = _chw(extra[:, 0])
                images["cam_right_wrist"] = _chw(extra[:, 1])
            return {
                "images": images,
                "state": env_states,
                "prompt": env_obs["task_descriptions"],
            }
        processed_obs = {
            "observation/image": env_obs["main_images"],
            "prompt": env_obs["task_descriptions"],
        }
        if "calvin" in self.config_name:
            processed_obs["observation/state_ee_pos"] = env_states[:, :3]
            processed_obs["observation/state_ee_rot"] = env_states[:, 3:6]
            processed_obs["observation/state_gripper"] = env_states[:, 6:7]
        else:
            processed_obs["observation/state"] = env_states
        wrist_images = env_obs.get("wrist_images")
        if wrist_images is not None:
            processed_obs["observation/wrist_image"] = wrist_images
        extra_view_images = env_obs.get("extra_view_images")
        if extra_view_images is not None:
            processed_obs["observation/extra_view_image"] = extra_view_images
        return processed_obs

    def input_transform(self, obs: dict, transpose: bool = False) -> dict:
        """Apply the openpi input pipeline per-sample then recombine into a batched dict.

        Two modes:

        * Rollout (``"prompt"`` key present) — runs the full pipeline including
          prompt tokenization; result has ``image``/``image_mask``/``state``/
          ``tokenized_prompt``/``tokenized_prompt_mask`` keys.
        * Train recompute (no ``"prompt"``; only ``observation/*`` and the cached
          ``tokenized_prompt`` keys present) — re-runs the pipeline using the
          cached tokens rather than re-tokenising every micro-batch.
        """
        self._ensure_wrappers()
        inputs = tree_map(lambda x: x, obs)
        first_process = "prompt" in inputs.keys()
        if first_process:
            inputs.pop("prompt")
        else:
            inputs = {k: inputs[k] for k in inputs.keys() if "/" in k}

        inputs = tree_map(_to_numpy, inputs)
        batch_size = next(v.shape[0] for v in inputs.values() if hasattr(v, "shape"))

        batch_samples = []
        for i in range(batch_size):
            sample = tree_map(lambda x: x[i], inputs)
            if transpose:
                sample = tree_map(
                    lambda x: (
                        x.transpose(1, 2, 0)
                        if isinstance(x, np.ndarray) and x.ndim == 3
                        else x
                    ),
                    sample,
                )
            if first_process:
                prompts = obs["prompt"]
                if isinstance(prompts, np.ndarray):
                    prompts = prompts.tolist()
                sample["prompt"] = prompts[i]
            else:
                # Pipeline still runs Tokenize, but the cached tokens below
                # overwrite its output — placeholder text is fine.
                sample["prompt"] = "xxxx"
            batch_samples.append(sample)

        with ThreadPoolExecutor(max_workers=min(len(batch_samples), 8)) as ex:
            transformed = list(ex.map(self._input_transform_fn, batch_samples))

        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if not first_process:
            recombined["tokenized_prompt"] = obs["tokenized_prompt"]
            recombined["tokenized_prompt_mask"] = obs["tokenized_prompt_mask"]
        return recombined

    def output_transform(self, outputs: dict) -> dict:
        """Apply the openpi output pipeline per-sample then recombine."""
        self._ensure_wrappers()
        batch_size = outputs["actions"].shape[0]
        transformed = []
        for i in range(batch_size):
            sample = tree_map(
                lambda x: _to_numpy(x[i]) if torch.is_tensor(x) else x[i],
                outputs,
            )
            sample = self._output_transform_fn(sample)
            transformed.append(sample)
        recombined = tree_map(
            lambda *xs: torch.from_numpy(np.asarray(xs).copy()),
            *transformed,
        )
        if self.action_chunk is not None:
            recombined["actions"] = recombined["actions"][:, : self.action_chunk]
        return recombined

    def _observation_dict_to_device(self, processed: dict) -> Observation:
        """Convert a per-key dict (from :meth:`input_transform`) into a device-resident :class:`Observation`."""
        device = self.device
        obs = Observation.from_dict(processed)

        def _move(x):
            return x.to(device) if isinstance(x, torch.Tensor) else x

        def _move_state(x):
            # The openpi Normalize stage runs in float64 (its norm_stats arrays
            # are float64); cast state back to float32 to match the legacy eval
            # processor (which did a final ``.float()``) and the model's compute
            # dtype. For pi05 the continuous state is unused (only the discrete
            # state tokens in the prompt are), but pi0 feeds it through
            # ``state_proj`` so float32 keeps the linear layer's dtype aligned.
            return (
                x.to(device=device, dtype=torch.float32)
                if isinstance(x, torch.Tensor)
                else x
            )

        return Observation(
            images={k: _move(v) for k, v in obs.images.items()},
            image_masks={k: _move(v) for k, v in obs.image_masks.items()},
            state=_move_state(obs.state),
            tokenized_prompt=_move(obs.tokenized_prompt),
            tokenized_prompt_mask=_move(obs.tokenized_prompt_mask),
            token_ar_mask=_move(obs.token_ar_mask),
            token_loss_mask=_move(obs.token_loss_mask),
            pcd_xyz=_move(obs.pcd_xyz),
        )

    # ------------------------------------------------------------------ rollout

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "eval",
        compute_values: bool = False,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Sample env actions via the deterministic Euler ODE sampler.

        Only ``mode="eval"`` is supported at the eval level — that path is
        shared by every transforms-pipeline variant (eval task + RL eval
        fallback). ``mode="train"`` is implemented in the RL subclass which
        overrides this method to add the chain-collecting SDE sampler.
        Calling with ``mode="train"`` on the eval class raises
        :class:`NotImplementedError` so an eval-only model loudly refuses to
        be used for on-policy rollouts.
        """
        del compute_values, kwargs  # accepted for call-site parity; eval ignores them
        if mode != "eval":
            raise NotImplementedError(
                f"{type(self).__name__} only supports predict_action_batch(mode='eval'); "
                "use the RL subclass (actor.model.openpi.task='rl') for train rollouts."
            )
        # openpi.transforms pipeline (eval / RL).
        repacked = self._repack_env_obs(env_obs)
        processed = self.input_transform(repacked, transpose=False)
        observation = self._observation_dict_to_device(processed)
        return self._predict_eval(observation, noise=noise, rng=rng)

    def _predict_eval(
        self,
        observation: Observation,
        *,
        noise: torch.Tensor | None,
        rng: torch.Generator | None,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Deterministic Euler ODE sampler shared by eval and the RL eval path.

        Returns the env-frame ``actions`` plus the rollout-side return dict
        with ``prev_logprobs=None`` / ``prev_values=None`` and a minimal
        ``forward_inputs`` (action + raw model_action) — exactly what
        :class:`huggingface_worker.HuggingFaceWorker.predict` expects from an
        eval call.
        """
        model_actions = self.model.sample_actions(
            observation, num_steps=self.num_steps, noise=noise, rng=rng
        )
        env_outputs = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )
        # openpi Unnormalize runs in float64; cast env actions back to float32 to
        # match the legacy eval processor's ``.astype(np.float32)`` contract (and
        # the action dtype the env/rollout worker expects).
        actions = self._to_rl_frame(env_outputs["actions"]).to(device=self.device, dtype=torch.float32)
        B = actions.shape[0]
        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": {
                "action": actions.reshape(B, -1).contiguous(),
                "model_action": model_actions.reshape(B, -1).contiguous(),
            },
        }
        return actions, result

    @torch.no_grad()
    def _to_rl_frame(self, actions: torch.Tensor) -> torch.Tensor:
        """env 动作帧 → RL 动作帧:按 RLT_ACTION_ENV_SCALE(逗号分隔,每维除数)缩放。
        A1 用 "1,1,1,1,1,1,100,1,1,1,1,1,1,100" 把夹爪 0-100 压到 0-1,使各维尺度可比。"""
        if self.config_name == "tianji_stage1":
            from rlinf.envs.realworld.tianji.tianji_contract import DEFAULT_CONTRACT
            scale = actions.new_tensor(DEFAULT_CONTRACT.action_scale)
            bias = actions.new_tensor(DEFAULT_CONTRACT.action_bias)
            encoded = (actions - bias) / scale
            if not torch.isfinite(encoded).all() or (encoded.abs() > 1.00001).any():
                raise ValueError("Stage1 target outside Stage2 absolute coordinate bounds")
            return encoded.clamp(-1, 1)
        if self.config_name == TIANJI_JOINT_CONFIG:
            # rtj joint Stage2: the joint contract's frame (SDK joint limits);
            # a predicted joint beyond its limit is clipped to it.
            from rlinf.envs.realworld.tianji.tianji_kinematics import (
                JOINT_ACTION_BIAS,
                JOINT_ACTION_SCALE,
            )

            encoded = (actions - actions.new_tensor(JOINT_ACTION_BIAS)) / actions.new_tensor(JOINT_ACTION_SCALE)
            if not torch.isfinite(encoded).all():
                raise ValueError("Stage1 joint target is not finite")
            return encoded.clamp(-1, 1)
        spec = os.environ.get("RLT_ACTION_ENV_SCALE", "")
        if not spec:
            return actions
        scale = torch.tensor([float(x) for x in spec.split(",")], device=actions.device, dtype=actions.dtype)
        bspec = os.environ.get("RLT_ACTION_ENV_BIAS", "")
        bias = torch.tensor([float(x) for x in bspec.split(",")], device=actions.device, dtype=actions.dtype) if bspec else torch.zeros_like(scale)
        return (actions - bias) / scale        # A1:关节按限位、夹爪按 0-100 归一到 [-1,1],与 tanh actor 匹配

    def _rtc_previous_joints(self, env_obs: dict[str, Any]):
        """``_rtc_previous_physical`` for the joint checkpoint (rtj joint Stage2).

        The in-flight rows are the env's 16D joint actions (RL frame): the VLA's
        own rows or a deployed actor's, so every request is guided. They are
        decoded by the joint contract and continued/padded in joint space.
        """
        valid = env_obs.get("rtc_prev_valid")
        actions = env_obs.get("rtc_prev_actions")
        if valid is None or actions is None:
            return None
        valid = torch.as_tensor(valid).reshape(-1).bool().cpu()
        actions = torch.as_tensor(actions).detach().double().cpu()
        if not valid.any() or actions.dim() != 3 or actions.shape[1] == 0:
            return None
        from rlinf.envs.realworld.tianji.tianji_kinematics import (
            JOINT_ACTION_BIAS,
            JOINT_ACTION_SCALE,
            JOINT_DIM,
        )

        if actions.shape[-1] != JOINT_DIM:
            raise ValueError(
                "the joint checkpoint's RTC needs the env's 16D joint actions "
                "(env.train.override_cfg.action_space: joint)"
            )
        horizon = int(self.model.action_horizon)
        rows = min(int(actions.shape[1]), horizon)
        physical = actions[:, :rows].numpy() * JOINT_ACTION_SCALE + JOINT_ACTION_BIAS
        extra = min(TIANJI_RTC_CONTINUATION_ROWS, horizon - rows)
        velocity = np.zeros_like(physical[:, -1:])
        if rows > 1:
            velocity[..., _TIANJI_JOINT_DIMS] = (
                physical[:, -1:, _TIANJI_JOINT_DIMS] - physical[:, -2:-1, _TIANJI_JOINT_DIMS]
            )
        steps = np.arange(1, extra + 1, dtype=np.float64)[None, :, None]
        guided = np.concatenate([physical, physical[:, -1:] + velocity * steps], axis=1)
        padded = np.concatenate(
            [guided, np.repeat(guided[:, -1:], horizon - rows - extra, axis=1)], axis=1
        )
        return padded.astype(np.float32), rows, extra, valid

    def _rtc_previous_physical(self, env_obs: dict[str, Any]):
        """In-flight actions of the executing chunk as physical targets.

        A Tianji stream-mode prefetch observation carries the steps of the
        current chunk that still execute while this request is in flight
        (``rtc_prev_actions``, RL frame). They are continued at their last
        velocity for TIANJI_RTC_CONTINUATION_ROWS rows, then padded to the
        model horizon, so the training input transforms map them into this
        request's model space (state-relative, normalized) like Stage1's RTC.
        Returns (physical [B, H, 20], in-flight rows, continuation rows, valid).
        """
        if self.config_name == TIANJI_JOINT_CONFIG:
            return self._rtc_previous_joints(env_obs)
        valid = env_obs.get("rtc_prev_valid")
        actions = env_obs.get("rtc_prev_actions")
        if valid is None or actions is None or self.config_name != "tianji_stage1":
            return None
        valid = torch.as_tensor(valid).reshape(-1).bool().cpu()
        actions = torch.as_tensor(actions).detach().float().cpu()
        if not valid.any() or actions.dim() != 3 or actions.shape[1] == 0:
            return None
        from rlinf.envs.realworld.tianji.tianji_contract import DEFAULT_CONTRACT

        horizon = int(self.model.action_horizon)
        rows = min(int(actions.shape[1]), horizon)
        physical = actions[:, :rows].numpy() * DEFAULT_CONTRACT.action_scale
        physical = physical + DEFAULT_CONTRACT.action_bias
        extra = min(TIANJI_RTC_CONTINUATION_ROWS, horizon - rows)
        velocity = np.zeros_like(physical[:, -1:])
        if rows > 1:
            velocity[..., _TIANJI_POSE_DIMS] = (
                physical[:, -1:, _TIANJI_POSE_DIMS] - physical[:, -2:-1, _TIANJI_POSE_DIMS]
            )
        steps = np.arange(1, extra + 1, dtype=np.float32)[None, :, None]
        guided = np.concatenate([physical, physical[:, -1:] + velocity * steps], axis=1)
        padded = np.concatenate(
            [guided, np.repeat(guided[:, -1:], horizon - rows - extra, axis=1)], axis=1
        )
        return padded.astype(np.float32), rows, extra, valid

    def _rtc_guidance(self, previous_model, rows, extra, valid, device):
        """RTC target and mask: the in-flight rows and the first continued row
        hard, the rest of the continuation soft (RLinf decaying weights)."""
        from rlinf.models.embodiment.openpi.rtc_guidance import build_rtc_target_and_mask

        target, mask = build_rtc_target_and_mask(
            prev_remaining=previous_model[:, : rows + extra].to(device, torch.float32),
            horizon=int(self.model.action_horizon),
            action_dim=int(self.model.action_dim),
            delay_steps=rows + min(extra, 1),
            device=device,
            dtype=torch.float32,
        )
        return target, mask * valid.to(device, torch.float32)[:, None, None]

    def extract_rlt_obs(self, env_obs: dict[str, Any]) -> dict[str, torch.Tensor]:
        """Extract the frozen Stage1 features consumed by the Stage2 RLT head.

        With an in-flight chunk in ``env_obs`` the reference chunk is sampled
        with RLinf RTC guidance: its first rows are pinned to those in-flight
        actions (hard rows), and so is the first executed row to their
        continuation, which then guides softly, so the next chunk continues
        the executed motion.
        """
        z_rl, ref_chunk = self.extract_vla_reference(env_obs)
        if self.config_name == TIANJI_JOINT_CONFIG:
            return self._joint_rlt_obs(env_obs, z_rl, ref_chunk)

        raw_proprio = self._select_configured_state(env_obs["states"])
        if "maniskill" in self.config_name.lower():
            state_dim = (
                raw_proprio.shape[-1]
                if hasattr(raw_proprio, "shape")
                else np.asarray(raw_proprio).shape[-1]
            )
            proprio = observation.state[..., :state_dim]
        else:
            proprio = raw_proprio
        if not torch.is_tensor(proprio):
            proprio = torch.as_tensor(proprio)

        return {
            "z_rl": z_rl,
            "proprio": proprio.to(device=z_rl.device, dtype=torch.float32),
            "ref_chunk": ref_chunk.to(device=z_rl.device, dtype=torch.float32),
        }

    def extract_vla_reference(
        self, env_obs: dict[str, Any], *, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The frozen VLA part of :meth:`extract_rlt_obs`: openpi preprocessing,
        prefix + RLT token, RTC-guided flow sampling, un-normalization.

        Returns ``(z_rl, ref_chunk)``: the RLT feature (float32, model device)
        and the reference chunk in the RL frame as :meth:`output_transform` /
        :meth:`_to_rl_frame` produce it. ``noise`` replaces the sampler's own
        ``torch.randn`` draw (the resident pi05_service receives it from the
        rollout, which drew it there); ``None`` draws it here as before.
        """
        self._require_rlt()
        repacked = self._repack_env_obs(env_obs)
        previous = self._rtc_previous_physical(env_obs)
        if previous is not None:
            repacked["actions"] = previous[0]
        processed = self.input_transform(repacked, transpose=False)
        previous_model = processed.pop("actions", None)
        observation = self._observation_dict_to_device(processed)

        if os.environ.get("TIANJI_FAST_INFER", "").strip() not in ("", "0"):
            # Opt-in (fast_infer.py): this block's computation from a CUDA graph;
            # unset = the eager block below, unchanged.
            from rlinf.models.embodiment.openpi_rlinf.fast_infer import rlt_forward

            z_rl, model_actions = rlt_forward(
                self, observation, previous, previous_model, noise=noise
            )
        else:
            prepared_observation = pi0_model_module.preprocess_observation(
                observation, train=False
            )
            prefix_output, prefix_mask, kv_cache = self.model.build_prefix_cache(
                prepared_observation
            )
            rlt_prefix_output, rlt_prefix_mask = self._select_rlt_prefix_embeddings(
                prefix_output, prefix_mask, prepared_observation.tokenized_prompt
            )
            z_rl = self._encode_rlt_flat(rlt_prefix_output, rlt_prefix_mask).to(
                dtype=torch.float32
            )

            rtc_target = rtc_mask = None
            if previous is not None:
                rtc_target, rtc_mask = self._rtc_guidance(
                    previous_model, *previous[1:], observation.state.device
                )
            model_actions = self._sample_actions_from_prefix_cache(
                prepared_observation,
                prefix_mask,
                kv_cache,
                noise=noise,
                rtc_target=rtc_target,
                rtc_mask=rtc_mask,
            )
        ref_chunk = self.output_transform(
            {"actions": model_actions, "state": observation.state}
        )["actions"]
        ref_chunk = self._to_rl_frame(torch.as_tensor(ref_chunk))
        return z_rl, ref_chunk

    def _joint_rlt_obs(self, env_obs, z_rl, ref_chunk):
        """RLT observation of the joint checkpoint (rtj joint Stage2).

        - ref_chunk: the VLA's joint chunk in the joint RL frame (the actor's
          reference and action space); ref_chunk_tcp: the same rows by FK in
          the TCP RL frame (stored for both-space data, never an input);
        - proprio (actor): the 16D joint state; rollout appends joint velocities
          and the stage one-hot;
        - critic_proprio: the 20D TCP state and the 18D TCP velocity of the
          measured joint velocities; rollout appends the stage one-hot.
        """
        from rlinf.envs.realworld.tianji.tianji_kinematics import (
            joint_rl_to_tcp_rl,
            tcp20_rate,
        )

        device = z_rl.device
        joint_state = torch.as_tensor(env_obs["joint_states"]).to(device=device, dtype=torch.float64)
        tcp_state = torch.as_tensor(env_obs["states"]).to(device=device, dtype=torch.float64)
        qvel = torch.as_tensor(env_obs["rlt_qvel"]).to(device=device, dtype=torch.float64)
        ref = ref_chunk.to(device=device, dtype=torch.float64)
        return {
            "z_rl": z_rl,
            "proprio": joint_state.to(torch.float32),
            "critic_proprio": torch.cat([tcp_state, tcp20_rate(joint_state, qvel)], dim=-1).to(torch.float32),
            "ref_chunk": ref.to(torch.float32),
            "ref_chunk_tcp": joint_rl_to_tcp_rl(ref).to(torch.float32),
        }

    def _sample_actions_from_prefix_cache(
        self,
        observation: Observation,
        prefix_mask: torch.Tensor,
        kv_cache: tuple,
        *,
        noise: torch.Tensor | None = None,
        rng: torch.Generator | None = None,
        rtc_target: torch.Tensor | None = None,
        rtc_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run Pi0's eval Euler sampler using an already-built prefix cache.

        ``rtc_target``/``rtc_mask`` (``build_rtc_target_and_mask``) apply
        RLinf RTC guidance to the predicted clean chunk ``x0 = x_t - t * v``
        and step with the velocity that reaches the blended estimate.
        """
        batch_size = observation.state.shape[0]
        device = observation.state.device
        if noise is None:
            noise = torch.randn(
                batch_size,
                self.model.action_horizon,
                self.model.action_dim,
                device=device,
                generator=rng,
            )
        guided = rtc_mask is not None and bool((rtc_mask > 0).any())

        x_t = noise
        dt = -1.0 / self.num_steps
        t = 1.0
        while t >= -dt / 2:
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.float32)
            suffix_out = self.model.run_suffix(
                observation, x_t, t_tensor, kv_cache, prefix_mask
            )
            v_t = self.model.velocity_from_suffix(suffix_out)
            if guided:
                x0 = x_t - t * v_t
                x0 = x0 + rtc_mask * (rtc_target - x0)
                v_t = (x_t - x0) / t
            x_t = x_t + dt * v_t
            t += dt
        return x_t
