# Copyright 2025 The RLinf Authors.
import copy
import glob
import importlib
import os
import sys
from typing import Optional, Union

import gym
import numpy as np
import torch
from omegaconf.omegaconf import OmegaConf

libero_type = os.environ.get("LIBERO_TYPE", "standard")

if libero_type == "pro":
    LIBERO_PKG_NAME = "liberopro"
    LIBERO_MAIN_MODULE_PATH = "liberopro.liberopro"
elif libero_type == "plus":
    LIBERO_PKG_NAME = "liberoplus"
    LIBERO_MAIN_MODULE_PATH = "liberoplus.liberoplus"
else:
    LIBERO_PKG_NAME = "libero"
    LIBERO_MAIN_MODULE_PATH = "libero.libero"

try:
    real_libero_pkg = importlib.import_module(LIBERO_PKG_NAME)
    real_libero_core = importlib.import_module(LIBERO_MAIN_MODULE_PATH)

    try:
        real_libero_benchmark = importlib.import_module(
            f"{LIBERO_MAIN_MODULE_PATH}.benchmark"
        )
    except ImportError:
        # Fallback for some versions
        real_libero_benchmark = importlib.import_module(f"{LIBERO_PKG_NAME}.benchmark")

    real_libero_envs = importlib.import_module(f"{LIBERO_MAIN_MODULE_PATH}.envs")

    if libero_type in ["pro", "plus"]:
        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs

    benchmark = real_libero_benchmark
    OffScreenRenderEnv = real_libero_envs.OffScreenRenderEnv

    if hasattr(real_libero_core, "get_libero_path"):
        get_libero_path = real_libero_core.get_libero_path
    else:
        try:
            real_libero_utils = importlib.import_module(
                f"{LIBERO_MAIN_MODULE_PATH}.utils"
            )
            get_libero_path = real_libero_utils.get_libero_path
        except (ImportError, AttributeError):

            def _fallback_get_libero_path(path_name):
                root = os.path.dirname(real_libero_core.__file__)
                return os.path.join(root, path_name)

            get_libero_path = _fallback_get_libero_path

except ImportError as e:
    raise ImportError(
        f"Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Check LIBERO_TYPE env var. Error: {e}"
    )


def apply_global_patches():
    custom_libero_path = os.environ.get("LIBERO_PATH", "")
    if custom_libero_path and os.path.exists(custom_libero_path):
        if custom_libero_path not in sys.path:
            sys.path.insert(0, custom_libero_path)
        clean_path = [p for p in sys.path if "/opt/libero" not in p]
        sys.path[:] = clean_path

    try:
        if getattr(torch.load, "_is_patched", False):
            pass
        else:
            _original_torch_load = torch.load

            def safe_torch_load(f, *args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False

                if isinstance(f, str) and not os.path.exists(f):
                    current_t = os.environ.get("LIBERO_TYPE", "standard")

                    if "/./" in f:
                        f = f.replace("/./", "/")

                    if (
                        current_t == "pro"
                        and "libero/libero" in f
                        and "liberopro" not in f
                    ):
                        new_f = f.replace("libero/libero", "liberopro/liberopro")
                        if os.path.exists(new_f):
                            f = new_f
                    elif (
                        current_t == "plus"
                        and "libero/libero" in f
                        and "liberoplus" not in f
                    ):
                        new_f = f.replace("libero/libero", "liberoplus/liberoplus")
                        if os.path.exists(new_f):
                            f = new_f

                return _original_torch_load(f, *args, **kwargs)

            safe_torch_load._is_patched = True
            torch.load = safe_torch_load
    except Exception as e:
        print(f"[LiberoEnv Patch]Failed to patch torch.load: {e}")

    try:
        current_type = os.environ.get("LIBERO_TYPE", "standard")

        if current_type == "pro":
            target_module_path = "liberopro.liberopro"
        elif current_type == "plus":
            target_module_path = "liberoplus.liberoplus"
        else:
            target_module_path = "libero.libero"

        libero_main = importlib.import_module(target_module_path)
        libero_objects = importlib.import_module(f"{target_module_path}.envs.objects")

        loaded_path = os.path.dirname(libero_main.__file__)

        bad_keys = {
            "white_white_porcelain_mug": "white_porcelain_mug",
            "white_yellow_porcelain_mug": "yellow_porcelain_mug",
            "white_red_porcelain_mug": "red_porcelain_mug",
        }
        if hasattr(libero_objects, "OBJECTS_DICT"):
            for bad_key, good_key in bad_keys.items():
                if bad_key not in libero_objects.OBJECTS_DICT:
                    if good_key in libero_objects.OBJECTS_DICT:
                        libero_objects.OBJECTS_DICT[bad_key] = (
                            libero_objects.OBJECTS_DICT[good_key]
                        )
                    elif "porcelain_mug" in libero_objects.OBJECTS_DICT:
                        libero_objects.OBJECTS_DICT[bad_key] = (
                            libero_objects.OBJECTS_DICT["porcelain_mug"]
                        )

        paths = {
            "assets": os.path.join(loaded_path, "assets"),
            "bddl_files": os.path.join(loaded_path, "bddl_files"),
            "init_states": os.path.join(loaded_path, "init_files"),
        }

        for k, p in paths.items():
            if not os.path.exists(p):
                if "liberopro" in loaded_path and "libero/libero" in p:
                    paths[k] = p.replace("libero/libero", "liberopro/liberopro")
                elif "liberoplus" in loaded_path and "libero/libero" in p:
                    paths[k] = p.replace("libero/libero", "liberoplus/liberoplus")

        os.environ["LIBERO_ASSET_ROOT"] = paths["assets"]
        os.environ["LIBERO_BDDL_PATH"] = paths["bddl_files"]
        os.environ["LIBERO_INIT_STATES_PATH"] = paths["init_states"]

        def force_local_path(path_name):
            if path_name in paths:
                return paths[path_name]
            return os.path.join(loaded_path, path_name)

        libero_main.get_libero_path = force_local_path

        try:
            bddl_utils = importlib.import_module(
                f"{target_module_path}.envs.bddl_utils"
            )
            original_get_problem_info = bddl_utils.get_problem_info

            def safe_get_problem_info(bddl_file_path):
                try:
                    return original_get_problem_info(bddl_file_path)
                except Exception:
                    return {
                        "domain_name": "unknown",
                        "problem_name": "unknown",
                        "language_instruction": "unknown task",
                    }

            bddl_utils.get_problem_info = safe_get_problem_info
        except ImportError:
            pass

    except Exception as e:
        print(f"[LiberoEnv Patch]Patching Error (pid={os.getpid()}): {e}")


apply_global_patches()

from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    put_info_on_image,
    quat2axisangle,
    save_rollout_video,
    tile_images,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


class LiberoEnv(gym.Env):
    def __init__(self, cfg, seed_offset, total_num_processes):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = self.cfg.num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.cfg.num_group
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)
        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset
        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        apply_global_patches()
        self.task_suite = get_benchmark_overridden(cfg.task_suite_name)()

        self._compute_total_num_group_envs()
        self.reset_state_ids_all = self.get_reset_state_ids_all()
        self.update_reset_state_ids()
        self._init_task_and_trial_ids()
        self._init_env()
        self.prev_step_reward = np.zeros(self.num_envs)
        self.use_rel_reward = cfg.use_rel_reward
        self._init_metrics()
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self.video_cfg = cfg.video_cfg
        self.video_cnt = 0
        self.render_images = []
        self.current_raw_obs = None

    def _init_env(self):
        env_fns = self.get_env_fns()
        self.env = ReconfigureSubprocEnv(env_fns)

    def get_env_fns(self):
        env_fn_params = self.get_env_fn_params()
        env_fns = []

        current_type_val = os.environ.get("LIBERO_TYPE", "standard")

        for env_fn_param in env_fn_params:

            def env_fn(param=env_fn_param, _type_val=current_type_val):
                # --- Worker Process Start ---
                import importlib
                import os
                import sys

                os.environ["LIBERO_TYPE"] = _type_val

                if _type_val in ["pro", "plus"]:
                    try:
                        if "libero" not in sys.modules:
                            if _type_val == "pro":
                                target_pkg_name = "liberopro"
                                target_core_name = "liberopro.liberopro"
                            else:  # plus
                                target_pkg_name = "liberoplus"
                                target_core_name = "liberoplus.liberoplus"

                            real_libero_pkg = importlib.import_module(target_pkg_name)
                            real_libero_core = importlib.import_module(target_core_name)

                            try:
                                real_libero_bench = importlib.import_module(
                                    f"{target_core_name}.benchmark"
                                )
                            except ImportError:
                                real_libero_bench = importlib.import_module(
                                    f"{target_pkg_name}.benchmark"
                                )

                            real_libero_envs = importlib.import_module(
                                f"{target_core_name}.envs"
                            )

                            sys.modules["libero"] = real_libero_pkg
                            sys.modules["libero.libero"] = real_libero_core
                            sys.modules["libero.libero.benchmark"] = real_libero_bench
                            sys.modules["libero.libero.envs"] = real_libero_envs
                    except ImportError:
                        pass

                apply_global_patches()

                seed = param.pop("seed")

                try:
                    if _type_val == "pro":
                        from liberopro.liberopro.envs import (
                            OffScreenRenderEnv as WorkerEnv,
                        )
                    elif _type_val == "plus":
                        from liberoplus.liberoplus.envs import (
                            OffScreenRenderEnv as WorkerEnv,
                        )
                    else:
                        from libero.libero.envs import OffScreenRenderEnv as WorkerEnv
                except ImportError:
                    try:
                        import libero.libero.envs as _le

                        WorkerEnv = _le.OffScreenRenderEnv
                    except:
                        raise ImportError(
                            f"Could not import OffScreenRenderEnv in worker (TYPE={_type_val})."
                        )

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)

        variant = os.environ.get(
            "LIBERO_TYPE", self.cfg.get("libero_variant", "standard")
        )
        plus_suffix = os.environ.get(
            "LIBERO_SUFFIX", self.cfg.get("perturbation_suffix", "")
        )
        pro_type = os.environ.get(
            "LIBERO_PERTURBATION", self.cfg.get("perturbation_type", None)
        )

        if plus_suffix:
            plus_suffix = plus_suffix.replace(".bddl", "")

        if env_idx is None or (len(env_idx) > 0 and env_idx[0] == 0):
            print(
                f"\n[LiberoEnv] Variant: {variant}, Suffix: '{plus_suffix}', Pro Type: '{pro_type}'"
            )

        if env_idx is None:
            env_idx = np.arange(self.cfg.num_envs)

        pro_folder_map = {
            "object": "object",
            "swap": "swap",
            "spatial": "swap",
            "language": "lan",
            "lan": "lan",
            "task": "task",
            "env": "env",
            "original": None,
            "all": "all",
        }

        bddl_root = get_libero_path("bddl_files")

        valid_params_cache = []
        valid_descriptions_cache = []
        valid_task_ids_cache = []
        current_task_descriptions = []

        for i, env_id in enumerate(env_idx):
            try:
                task_idx = self.task_ids[env_id]
                task = self.task_suite.get_task(task_idx)
            except IndexError:
                if len(valid_params_cache) > 0:
                    env_fn_params.append(valid_params_cache[-1])
                    current_task_descriptions.append(valid_descriptions_cache[-1])
                    self.task_ids[env_id] = valid_task_ids_cache[-1]
                continue

            folder_name = task.problem_folder
            file_name = task.bddl_file

            final_path = None
            original_path = os.path.join(bddl_root, folder_name, file_name)

            # --- Logic for Pro ---
            if variant == "pro" and pro_type in pro_folder_map:
                suffix = pro_folder_map[pro_type]
                if suffix:
                    perturbed_folder = f"{folder_name}_{suffix}"
                    target_path = os.path.join(bddl_root, perturbed_folder, file_name)
                    if os.path.exists(target_path):
                        final_path = target_path

            # --- Logic for Plus ---
            elif variant == "plus" and plus_suffix:
                base_name = file_name.replace(".bddl", "")
                clean_name = base_name

                markers = [
                    "_sample",
                    "_view",
                    "_initstate",
                    "_noise",
                    "_light",
                    "_table",
                    "_add",
                    "_lan",
                    "_language",
                    "_copy",
                ]

                for marker in markers:
                    if marker in clean_name:
                        clean_name = clean_name.split(marker)[0]
                        break

                if plus_suffix.lower() == "all":
                    pattern = f"{clean_name}*.bddl"
                else:
                    pattern = f"{clean_name}*{plus_suffix}.bddl"

                search_full_path = os.path.join(bddl_root, folder_name, pattern)

                try:
                    candidates = glob.glob(search_full_path)
                    candidates.sort()

                    if len(candidates) > 0:
                        if plus_suffix.lower() == "all":
                            final_path = candidates[i % len(candidates)]
                        else:
                            final_path = candidates[0]
                except Exception as e:
                    print(f"[LiberoEnv] Glob search error: {e}")

            if final_path is None:
                if os.path.exists(original_path):
                    final_path = original_path
            if final_path is None or not os.path.exists(final_path):
                if len(valid_params_cache) > 0:
                    env_fn_params.append(valid_params_cache[-1])
                    current_task_descriptions.append(valid_descriptions_cache[-1])
                    self.task_ids[env_id] = valid_task_ids_cache[-1]
                    continue
                else:
                    print(f"\n[LiberoEnv] SKIPPING Task: {file_name}")
                    if variant == "plus":
                        print(
                            f"   Target Pattern: {pattern if 'pattern' in locals() else 'N/A'}"
                        )
                    continue

            current_param = {
                **base_env_args,
                "bddl_file_name": final_path,
                "seed": self.seed,
            }

            try:
                desc = task.language
            except:
                desc = "unknown task"

            env_fn_params.append(current_param)
            current_task_descriptions.append(desc)

            valid_params_cache.append(current_param)
            valid_descriptions_cache.append(desc)
            valid_task_ids_cache.append(task_idx)

        if len(env_fn_params) == 0:
            raise RuntimeError(
                "[LiberoEnv]  CRITICAL: No valid BDDL files found! Please check LIBERO_PATH and SUFFIX."
            )

        while len(env_fn_params) < len(env_idx):
            if len(valid_params_cache) > 0:
                env_fn_params.append(valid_params_cache[-1])
                current_task_descriptions.append(valid_descriptions_cache[-1])
                target_env_id = env_idx[len(env_fn_params) - 1]
                self.task_ids[target_env_id] = valid_task_ids_cache[-1]
            else:
                break

        self.task_descriptions = current_task_descriptions
        return env_fn_params

    def _compute_total_num_group_envs(self):
        self.total_num_group_envs = 0
        self.trial_id_bins = []
        for task_id in range(self.task_suite.get_num_tasks()):
            task_num_trials = len(self.task_suite.get_task_init_states(task_id))
            self.trial_id_bins.append(task_num_trials)
            self.total_num_group_envs += task_num_trials
        self.cumsum_trial_id_bins = np.cumsum(self.trial_id_bins)

    def update_reset_state_ids(self):
        if self.cfg.only_eval or self.cfg.use_ordered_reset_state_ids:
            reset_state_ids = self._get_ordered_reset_state_ids(self.num_group)
        else:
            reset_state_ids = self._get_random_reset_state_ids(self.num_group)
        self.reset_state_ids = reset_state_ids.repeat(self.group_size)

    def _init_task_and_trial_ids(self):
        self.task_ids, self.trial_ids = (
            self._get_task_and_trial_ids_from_reset_state_ids(self.reset_state_ids)
        )

    def _get_random_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (num_reset_states,), dtype=int
            )
        else:
            reset_state_ids = self._generator.integers(
                low=0, high=self.total_num_group_envs, size=(num_reset_states,)
            )
        return reset_state_ids

    def get_reset_state_ids_all(self):
        reset_state_ids = np.arange(self.total_num_group_envs)
        valid_size = len(reset_state_ids) - (
            len(reset_state_ids) % self.total_num_processes
        )
        self._generator_ordered.shuffle(reset_state_ids)
        reset_state_ids = reset_state_ids[:valid_size]
        reset_state_ids = reset_state_ids.reshape(self.total_num_processes, -1)
        return reset_state_ids

    def _get_ordered_reset_state_ids(self, num_reset_states):
        if self.specific_reset_id is not None:
            reset_state_ids = self.specific_reset_id * np.ones(
                (self.num_group,), dtype=int
            )
        else:
            if self.start_idx + num_reset_states > len(self.reset_state_ids_all[0]):
                self.reset_state_ids_all = self.get_reset_state_ids_all()
                self.start_idx = 0
            reset_state_ids = self.reset_state_ids_all[self.seed_offset][
                self.start_idx : self.start_idx + num_reset_states
            ]
            self.start_idx = self.start_idx + num_reset_states
        return reset_state_ids

    def _get_task_and_trial_ids_from_reset_state_ids(self, reset_state_ids):
        task_ids = []
        trial_ids = []
        for reset_state_id in reset_state_ids:
            start_pivot = 0
            for task_id, end_pivot in enumerate(self.cumsum_trial_id_bins):
                if reset_state_id < end_pivot and reset_state_id >= start_pivot:
                    task_ids.append(task_id)
                    trial_ids.append(reset_state_id - start_pivot)
                    break
                start_pivot = end_pivot
        return np.array(task_ids), np.array(trial_ids)

    def _get_reset_states(self, env_idx):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        init_state = [
            self.task_suite.get_task_init_states(self.task_ids[env_id])[
                self.trial_ids[env_id]
            ]
            for env_id in env_idx
        ]
        return init_state

    @property
    def elapsed_steps(self):
        return self._elapsed_steps

    @property
    def info_logging_keys(self):
        return []

    @property
    def is_start(self):
        return self._is_start

    @is_start.setter
    def is_start(self, value):
        self._is_start = value

    def _init_metrics(self):
        self.success_once = np.zeros(self.num_envs, dtype=bool)
        self.fail_once = np.zeros(self.num_envs, dtype=bool)
        self.returns = np.zeros(self.num_envs)

    def _reset_metrics(self, env_idx=None):
        if env_idx is not None:
            mask = np.zeros(self.num_envs, dtype=bool)
            mask[env_idx] = True
            self.prev_step_reward[mask] = 0.0
            self.success_once[mask] = False
            self.fail_once[mask] = False
            self.returns[mask] = 0
            self._elapsed_steps[env_idx] = 0
        else:
            self.prev_step_reward[:] = 0
            self.success_once[:] = False
            self.fail_once[:] = False
            self.returns[:] = 0.0
            self._elapsed_steps[:] = 0

    def _record_metrics(self, step_reward, terminations, infos):
        episode_info = {}
        self.returns += step_reward
        self.success_once = self.success_once | terminations
        episode_info["success_once"] = self.success_once.copy()
        episode_info["return"] = self.returns.copy()
        episode_info["episode_len"] = self.elapsed_steps.copy()
        episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
        infos["episode"] = to_tensor(episode_info)
        return infos

    def _extract_image_and_state(self, obs):
        if self.cfg.get("use_wrist_image", False):
            return {
                "full_image": get_libero_image(obs),
                "wrist_image": get_libero_wrist_image(obs),
                "state": np.concatenate(
                    [
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ]
                ),
            }
        else:
            return {
                "full_image": get_libero_image(obs),
                "state": np.concatenate(
                    [
                        obs["robot0_eef_pos"],
                        quat2axisangle(obs["robot0_eef_quat"]),
                        obs["robot0_gripper_qpos"],
                    ]
                ),
            }

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        obs = {
            "images_and_states": to_tensor(
                list_of_dict_to_dict_of_list(images_and_states_list)
            ),
            "task_descriptions": self.task_descriptions,
        }
        return obs

    def _reconfigure(self, reset_state_ids, env_idx):
        reconfig_env_idx = []
        task_ids, trial_ids = self._get_task_and_trial_ids_from_reset_state_ids(
            reset_state_ids
        )
        for j, env_id in enumerate(env_idx):
            if self.task_ids[env_id] != task_ids[j]:
                reconfig_env_idx.append(env_id)
            self.task_ids[env_id] = task_ids[j]
            self.trial_ids[env_id] = trial_ids[j]
        if reconfig_env_idx:
            env_fn_params = self.get_env_fn_params(reconfig_env_idx)
            self.env.reconfigure_env_fns(env_fn_params, reconfig_env_idx)
        self.env.seed(self.seed * len(env_idx))
        self.env.reset(id=env_idx)
        variant = os.environ.get("LIBERO_TYPE", "standard")
        if variant in ["plus", "pro"]:
            pass
        else:
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
        options: Optional[dict] = {},
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            zero_actions[:, -1] = -1
            raw_obs, _reward, terminations, info_lists = self.env.step(
                zero_actions, env_idx
            )
        if self.current_raw_obs is None:
            self.current_raw_obs = [None] * self.num_envs
        for i, idx in enumerate(env_idx):
            self.current_raw_obs[idx] = raw_obs[i]

        obs = self._wrap_obs(self.current_raw_obs)
        self._reset_metrics(env_idx)
        infos = {}
        return obs, infos

    def step(self, actions=None, auto_reset=True):
        if actions is None:
            assert self._is_start, "Actions must be provided after the first reset."
        if self.is_start:
            obs, infos = self.reset(
                reset_state_ids=self.reset_state_ids
                if self.use_fixed_reset_state_ids
                else None
            )
            self._is_start = False
            terminations = np.zeros(self.num_envs, dtype=bool)
            truncations = np.zeros(self.num_envs, dtype=bool)

            return obs, None, to_tensor(terminations), to_tensor(truncations), infos

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

        if self.video_cfg.save_video:
            plot_infos = {
                "rewards": step_reward,
                "terminations": terminations,
                "task": self.task_descriptions,
            }
            self.add_new_frames(raw_obs, plot_infos)

        infos = self._record_metrics(step_reward, terminations, infos)
        if self.ignore_terminations:
            infos["episode"]["success_at_end"] = to_tensor(terminations)
            terminations[:] = False

        dones = terminations | truncations
        _auto_reset = auto_reset and self.auto_reset
        if dones.any() and _auto_reset:
            obs, infos = self._handle_auto_reset(dones, obs, infos)
        return (
            obs,
            to_tensor(step_reward),
            to_tensor(terminations),
            to_tensor(truncations),
            infos,
        )

    def chunk_step(self, chunk_actions):
        chunk_size = chunk_actions.shape[1]
        chunk_rewards = []
        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)
        raw_chunk_terminations = torch.stack(raw_chunk_terminations, dim=1)
        raw_chunk_truncations = torch.stack(raw_chunk_truncations, dim=1)

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            extracted_obs, infos = self._handle_auto_reset(
                past_dones.cpu().numpy(), extracted_obs, infos
            )

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_chunk_terminations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations = torch.zeros_like(raw_chunk_truncations)
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_chunk_terminations.clone()
            chunk_truncations = raw_chunk_truncations.clone()
        return (
            extracted_obs,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        infos["final_observation"] = final_obs
        infos["final_info"] = final_info
        infos["_final_info"] = dones
        infos["_final_observation"] = dones
        infos["_elapsed_steps"] = dones
        return obs, infos

    def _calc_step_reward(self, terminations):
        reward = self.cfg.reward_coef * terminations
        reward_diff = reward - self.prev_step_reward
        self.prev_step_reward = reward

        if self.use_rel_reward:
            return reward_diff
        else:
            return reward

    def add_new_frames(self, raw_obs, plot_infos):
        images = []
        for env_id, raw_single_obs in enumerate(raw_obs):
            info_item = {
                k: v if np.size(v) == 1 else v[env_id] for k, v in plot_infos.items()
            }
            img = raw_single_obs["agentview_image"][::-1]
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir=None):
        output_dir = os.path.join(self.video_cfg.video_base_dir, f"seed_{self.seed}")
        if video_sub_dir is not None:
            output_dir = os.path.join(output_dir, f"{video_sub_dir}")
        save_rollout_video(
            self.render_images,
            output_dir=output_dir,
            video_name=f"{self.video_cnt}",
        )
        self.video_cnt += 1
        self.render_images = []
