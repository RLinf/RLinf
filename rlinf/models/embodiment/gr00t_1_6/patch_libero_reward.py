"""
Monkey-patch for LiberoEnv._calc_step_reward to provide a denser reward signal for PPO.

This file patches:
1. LiberoEnv._calc_step_reward - adds per-step penalty 
2. EnvWorker.init_worker - ensures env processes also get the reward patch
3. rlinf.envs.get_env_cls - auto-applies patch when LIBERO env is requested
"""

import sys
import types

_PATCHED = False


def _patch_libero_calc_reward():
    """Patch LiberoEnv._calc_step_reward to add per-step penalty."""
    import rlinf.envs.libero.libero_env as le
    import numpy as np

    def _patched_calc(self, terminations):
        # Original reward: terminations * reward_coef (config default is 1.0)
        # Our patch: moderate per-step penalty + strong success bonus.
        step_penalty = -1 if self.use_step_penalty else 0
        term_np = np.asarray(terminations, dtype=np.float32)
        # 240 steps * -0.01 = -2.4, success = +5.0 => net +2.6, clearly better
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


def _patch_gr00t_explore_noise():
    """
    Patch GR00T to add exploration noise during rollout.

    Strategy:
    1. Set flow_sde noise config in __init__ 
    2. Patch sample_mean_var_val to always use mode="train"
    3. Patch rollout worker's predict to add Gaussian noise to actions
       (belt-and-suspenders approach)
    """
    try:
        import rlinf.models.embodiment.gr00t_1_6.gr00t_action_model as gam
        import torch
        import numpy as np

        # Fix 1: Set stronger noise config
        _orig_init = gam.FlowMatchingActionHeadForRLActionPrediction.__init__

        def _patched_init(self, *args, **kwargs):
            _orig_init(self, *args, **kwargs)
            self.rl_config["noise_method"] = "flow_sde"
            self.rl_config["noise_level"] = 5.0
            self.rl_config["noise_anneal"] = False

        gam.FlowMatchingActionHeadForRLActionPrediction.__init__ = _patched_init

        # Fix 2: Always use train mode in sample_mean_var_val
        _orig_sample = gam.FlowMatchingActionHeadForRLActionPrediction.sample_mean_var_val

        def _patched_sample(self, *args, **kwargs):
            kwargs["mode"] = "train"
            return _orig_sample(self, *args, **kwargs)

        gam.FlowMatchingActionHeadForRLActionPrediction.sample_mean_var_val = _patched_sample

        # Fix 3: Patch rollout predict to add Gaussian action noise
        try:
            from rlinf.workers.rollout.hf import huggingface_worker as hfw
            _orig_predict = hfw.MultiStepRolloutWorker.predict

            def _patched_predict(self, env_obs, mode="train"):
                actions, result = _orig_predict(self, env_obs, mode)
                if mode == "train":
                    # Add Gaussian noise to actions for exploration
                    # Action range is approximately [-1, 1] for LIBERO
                    noise_scale = 0.3
                    noise = torch.randn_like(actions) * noise_scale
                    actions = actions + noise
                    # Clip to valid range
                    actions = torch.clamp(actions, -1.0, 1.0)
                return actions, result

            hfw.MultiStepRolloutWorker.predict = _patched_predict
            print("[GR00T patch] Rollout predict patched: Gaussian noise 0.3 on actions")
        except Exception as e:
            print(f"[GR00T patch] Failed to patch rollout predict: {e}")

        print("[GR00T patch] GR00T explore noise: train mode for ALL steps, noise_level=5.0, action_noise=0.3")
    except Exception as e:
        print(f"[GR00T patch] Failed to patch GR00T explore noise: {e}")


def _patch_worker_init():
    """Patch EnvWorker.init_worker to apply reward patch before env creation."""
    try:
        from rlinf.scheduler import Worker
        import numpy as np

        # Find EnvWorker in subclasses
        for subcls in Worker.__subclasses__():
            if subcls.__name__ == "EnvWorker":
                orig_init = subcls.init_worker

                def _patched_init_worker(self):
                    # Apply the reward patch before setting up envs
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
        if str(env_type) == "libero" or (hasattr(env_type, "value") and env_type.value == "libero"):
            try:
                _patch_libero_calc_reward()
            except Exception:
                pass
        return result

    env_module.get_env_cls = _patched


def patch_libero_reward():
    global _PATCHED
    if _PATCHED:
        return
    _PATCHED = True

    try:
        _patch_libero_calc_reward()
        print("[GR00T patch] LiberoEnv._calc_step_reward patched: per-step -0.01, success x5")
    except Exception as e:
        print(f"[GR00T patch] Failed direct patch: {e}")

    try:
        _patch_gr00t_explore_noise()
    except Exception as e:
        print(f"[GR00T patch] Failed noise patch: {e}")

    try:
        _patch_get_env_cls()
        print("[GR00T patch] get_env_cls patched: auto-applies reward patch on LIBERO env")
    except Exception as e:
        print(f"[GR00T patch] Failed get_env_cls patch: {e}")

    try:
        _patch_worker_init()
    except Exception as e:
        print(f"[GR00T patch] Failed worker init patch: {e}")
