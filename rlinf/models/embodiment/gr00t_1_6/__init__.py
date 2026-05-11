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

import sys
import types
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

# Monkey-patch: inject a stub for rlinf.envs.libero.asset_paths so that
# env workers can import it even when the file is absent from the container.
# This avoids modifying the official rlinf/envs/libero/libero_env.py.
_OLD_ASSET_PATHS = sys.modules.get("rlinf.envs.libero.asset_paths")
if _OLD_ASSET_PATHS is None:
    _stub = types.ModuleType("rlinf.envs.libero.asset_paths")

    def _noop(*args, **kwargs):
        pass

    _stub.apply_standard_libero_env_vars = _noop
    sys.modules["rlinf.envs.libero.asset_paths"] = _stub


def get_model(cfg: DictConfig, torch_dtype=torch.bfloat16):
    from gr00t.configs.model.gr00t_n1d6 import Gr00tN1d6Config
    from gr00t.model.gr00t_n1d6.gr00t_n1d6 import Gr00tN1d6
    from transformers import AutoConfig, AutoModel

    AutoConfig.register("Gr00tN1d6", Gr00tN1d6Config)
    AutoModel.register(Gr00tN1d6Config, Gr00tN1d6)
    print(
        "Successfully registered custom architecture Gr00tN1d6, authentication passed!"
    )
    import rlinf.hybrid_engines.fsdp.strategy.fsdp as fsdp_strategy

    if not hasattr(fsdp_strategy, "_is_gr00t_patched"):
        orig_policy = fsdp_strategy.get_fsdp_wrap_policy

        def custom_fsdp_wrap_policy(
            module, config=None, is_lora=False, model_type=None
        ):
            import functools

            from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

            target_keywords = [
                "DecoderLayer",
                "EncoderLayer",
                "DiTBlock",
                "NoiseNet",
                "ValueHead",
                "ActionHead",
                "Timestep",
            ]
            found_classes = set()
            for name, mod in module.named_modules():
                cname = mod.__class__.__name__
                if any(key in cname for key in target_keywords):
                    found_classes.add(mod.__class__)

            if found_classes:
                print(f"\n  FSDP Slicer: {[c.__name__ for c in found_classes]}\n")
                return functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=found_classes
                )

            return orig_policy(module, config, is_lora, model_type)

        fsdp_strategy.get_fsdp_wrap_policy = custom_fsdp_wrap_policy
        fsdp_strategy._is_gr00t_patched = True
    from rlinf.utils.patcher import Patcher

    Patcher.clear()
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EmbodimentTag",
        "rlinf.models.embodiment.gr00t_1_6.embodiment_tags.EmbodimentTag",
    )
    Patcher.add_patch(
        "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        "rlinf.models.embodiment.gr00t_1_6.embodiment_tags.EMBODIMENT_TAG_MAPPING",
    )
    Patcher.apply()

    from gr00t.data.embodiment_tags import EmbodimentTag

    from rlinf.models.embodiment.gr00t_1_6.gr00t_action_model import (
        GR00T_N1_6_ForRLActionPrediction,
    )
    from rlinf.models.embodiment.gr00t_1_6.utils import replace_dropout_with_identity

    use_official_libero_panda = bool(
        OmegaConf.select(cfg, "use_official_libero_panda", default=False)
    )

    if cfg.embodiment_tag == "libero_panda":
        emb_tag = (
            EmbodimentTag.LIBERO_PANDA
            if use_official_libero_panda
            else EmbodimentTag.ROBOCASA_PANDA_OMRON
        )
    elif cfg.embodiment_tag in [
        "libero_franka",
        "isaaclab_franka",
        "maniskill_widowx",
        "robocasa_panda_omron",
    ]:
        emb_tag = EmbodimentTag.ROBOCASA_PANDA_OMRON
    elif cfg.embodiment_tag == "gr1":
        emb_tag = EmbodimentTag.GR1
    elif cfg.embodiment_tag == "behavior_r1_pro":
        emb_tag = EmbodimentTag.BEHAVIOR_R1_PRO
    else:
        raise ValueError(
            f"Invalid or unsupported embodiment tag: {cfg.embodiment_tag}. "
            f"Supported tags are: ['behavior_r1_pro', 'gr1', 'robocasa_panda_omron', 'libero_panda']."
        )

    model_path = Path(cfg.model_path)
    if not model_path.exists():
        raise FileNotFoundError(f"Model path does not exist: {model_path}")

    if cfg.get("model_type") == "gr00t_1_6_sft":
        from .gr00t_16_sft_model import GR00T_1_6_SFT_Model

        model_cls = GR00T_1_6_SFT_Model
    else:
        model_cls = GR00T_N1_6_ForRLActionPrediction

    processor_path = OmegaConf.select(cfg, "processor_path", default=None)

    model = model_cls.from_pretrained(
        local_model_path=str(model_path),
        pretrained_model_name_or_path=str(model_path),
        # model_path,
        torch_dtype=torch_dtype,
        embodiment_tag=emb_tag,
        denoising_steps=cfg.denoising_steps,
        output_action_chunks=cfg.num_action_chunks,
        obs_converter_type=cfg.obs_converter_type,
        tune_visual=False,
        tune_llm=False,
        rl_head_config=cfg.rl_head_config,
        processor_path=processor_path,
    )

    model.to(torch_dtype)
    if cfg.rl_head_config.add_value_head and hasattr(model.action_head, "value_head"):
        # reinitialize the value head after model loading
        model.action_head.value_head._init_weights()

    if cfg.rl_head_config.disable_dropout:
        replace_dropout_with_identity(model)

    return model


_RLINF_GLOBAL_PATCHED = False


def _patch_libero_calc_reward():
    """Patch LiberoEnv._calc_step_reward to add per-step penalty."""
    import numpy as np

    import rlinf.envs.libero.libero_env as le

    def _patched_calc(self, terminations):
        step_penalty = -1 if self.use_step_penalty else 0
        term_np = np.asarray(terminations, dtype=np.float32)
        success_bonus = self.cfg.reward_coef * 5.0
        termination_bonus = success_bonus * term_np
        reward = step_penalty + termination_bonus

        if self.use_rel_reward:
            reward_diff = reward - self.prev_step_reward
            self.prev_step_reward = reward
            return reward_diff
        else:
            if not self.use_step_penalty:
                non_term_mask = ~(term_np > 0)
                reward = np.asarray(reward, dtype=np.float32)
                reward = reward - 0.01 * non_term_mask
            return reward

    le.LiberoEnv._calc_step_reward = _patched_calc


def _patch_rollout_worker_predict():
    """Patch Rollout worker to add Gaussian noise to actions."""
    try:
        import torch

        from rlinf.workers.rollout.hf import huggingface_worker as hfw

        _orig_predict = hfw.MultiStepRolloutWorker.predict

        def _patched_predict(self, env_obs, mode="train"):
            actions, result = _orig_predict(self, env_obs, mode)
            if mode == "train":
                # Add Gaussian noise to actions for exploration
                noise_scale = 0.3
                noise = torch.randn_like(actions) * noise_scale
                actions = actions + noise
                actions = torch.clamp(actions, -1.0, 1.0)
            return actions, result

        hfw.MultiStepRolloutWorker.predict = _patched_predict
        print("[GR00T patch] Rollout predict patched: Gaussian noise 0.3 on actions")
    except Exception as e:
        print(f"[GR00T patch] Failed to patch rollout predict: {e}")


def _patch_worker_init():
    """Patch EnvWorker.init_worker to apply reward patch before env creation."""
    try:
        from rlinf.scheduler import Worker

        for subcls in Worker.__subclasses__():
            if subcls.__name__ == "EnvWorker":
                orig_init = subcls.init_worker

                def _patched_init_worker(self):
                    try:
                        _patch_libero_calc_reward()
                    except Exception:
                        pass
                    return orig_init(self)

                subcls.init_worker = _patched_init_worker
                print("[GR00T patch] EnvWorker.init_worker patched to apply reward fix")
                return
        print("[GR00T patch] EnvWorker not found in Worker subclasses")
    except Exception as e:
        print(f"[GR00T patch] Failed to patch EnvWorker.init_worker: {e}")


def _patch_get_env_cls():
    """Patch get_env_cls for good measure."""
    import rlinf.envs as env_module

    _orig = env_module.get_env_cls

    def _patched(env_type, env_cfg=None):
        result = _orig(env_type, env_cfg)
        if str(env_type) == "libero" or (
            hasattr(env_type, "value") and env_type.value == "libero"
        ):
            try:
                _patch_libero_calc_reward()
            except Exception:
                pass
        return result

    env_module.get_env_cls = _patched


def apply_global_rlinf_patches():
    global _RLINF_GLOBAL_PATCHED
    if _RLINF_GLOBAL_PATCHED:
        return
    _RLINF_GLOBAL_PATCHED = True

    # Only apply rollout predict patch at module level — it does not import
    # tensorflow and is needed by the RolloutGroup.  Env-related patches
    # (_patch_libero_calc_reward, _patch_get_env_cls) import tensorflow and
    # will crash Ray workers that don't need it (e.g. RolloutGroup, ActorGroup).
    # They are deferred and applied inside EnvWorker.init_worker via
    # _patch_worker_init below.

    try:
        _patch_rollout_worker_predict()
    except Exception as e:
        print(f"[GR00T patch] Failed noise patch (rollout): {e}")

    try:
        _patch_worker_init()
    except Exception as e:
        print(f"[GR00T patch] Failed worker init patch: {e}")


apply_global_rlinf_patches()
