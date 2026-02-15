# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Dynamic Module Import Logic
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
        real_libero_benchmark = importlib.import_module(f"{LIBERO_MAIN_MODULE_PATH}.benchmark")
    except ImportError:
        try:
            real_libero_benchmark = importlib.import_module(f"{LIBERO_PKG_NAME}.benchmark")
        except ImportError:
            if libero_type == "plus":
                 real_libero_benchmark = importlib.import_module("liberoplus.liberoplus.benchmark")
            else:
                raise

    try:
        real_libero_envs = importlib.import_module(f"{LIBERO_MAIN_MODULE_PATH}.envs")
    except ImportError:
        real_libero_envs = importlib.import_module(f"{LIBERO_PKG_NAME}.envs")

    if libero_type in ["pro", "plus"]:
        sys.modules["libero"] = real_libero_pkg
        sys.modules["libero.libero"] = real_libero_core
        sys.modules["libero.libero.benchmark"] = real_libero_benchmark
        sys.modules["libero.libero.envs"] = real_libero_envs

    Benchmark = real_libero_benchmark.Benchmark
    OffScreenRenderEnv = real_libero_envs.OffScreenRenderEnv

    if hasattr(real_libero_core, "get_libero_path"):
        get_libero_path = real_libero_core.get_libero_path
    else:
        try:
            real_libero_utils = importlib.import_module(f"{LIBERO_MAIN_MODULE_PATH}.utils")
            get_libero_path = real_libero_utils.get_libero_path
        except (ImportError, AttributeError):
            def _fallback_get_libero_path(path_name):
                if hasattr(real_libero_core, "__path__"):
                    root = list(real_libero_core.__path__)[0]
                else:
                    root = os.path.dirname(real_libero_core.__file__)
                return os.path.join(root, path_name)
            get_libero_path = _fallback_get_libero_path

except ImportError as e:
    raise ImportError(f"Failed to import '{LIBERO_MAIN_MODULE_PATH}'. Check LIBERO_TYPE env var. Error: {e}")


# Global Patch Function for Fixing torch.load and Paths
def apply_global_patches():
    custom_libero_path = os.environ.get("LIBERO_PATH", "")
    if custom_libero_path and os.path.exists(custom_libero_path):
        if custom_libero_path not in sys.path:
            sys.path.insert(0, custom_libero_path)
        sys.path[:] = [p for p in sys.path if "/opt/libero" not in p]

    try:
        if not getattr(torch.load, "_is_patched", False):
            _original_torch_load = torch.load
            def safe_torch_load(f, *args, **kwargs):
                if "weights_only" not in kwargs:
                    kwargs["weights_only"] = False
                if isinstance(f, str) and not os.path.exists(f):
                    current_t = os.environ.get("LIBERO_TYPE", "standard")
                    if "/./" in f: f = f.replace("/./", "/")
                    for t_name in ["pro", "plus"]:
                        if current_t == t_name and "libero/libero" in f:
                            new_f = f.replace("libero/libero", f"libero{t_name}/libero{t_name}")
                            if os.path.exists(new_f): f = new_f
                return _original_torch_load(f, *args, **kwargs)
            safe_torch_load._is_patched = True
            torch.load = safe_torch_load
    except Exception as e:
        print(f"[LiberoEnv Patch] Failed to patch torch.load: {e}")

    try:
        current_type = os.environ.get("LIBERO_TYPE", "standard")
        if current_type == "pro": target_module_path = "liberopro.liberopro"
        elif current_type == "plus": target_module_path = "liberoplus.liberoplus"
        else: target_module_path = "libero.libero"

        libero_main = importlib.import_module(target_module_path)
        libero_objects = importlib.import_module(f"{target_module_path}.envs.objects")
        loaded_path = os.path.dirname(libero_main.__file__)

        if hasattr(libero_objects, "OBJECTS_DICT"):
            bad_keys = {
                "white_white_porcelain_mug": "white_porcelain_mug", 
                "white_yellow_porcelain_mug": "yellow_porcelain_mug", 
                "white_red_porcelain_mug": "red_porcelain_mug"
            }
            for b_key, g_key in bad_keys.items():
                if b_key not in libero_objects.OBJECTS_DICT:
                    if g_key in libero_objects.OBJECTS_DICT:
                        libero_objects.OBJECTS_DICT[b_key] = libero_objects.OBJECTS_DICT[g_key]
                    elif "porcelain_mug" in libero_objects.OBJECTS_DICT:
                        libero_objects.OBJECTS_DICT[b_key] = libero_objects.OBJECTS_DICT["porcelain_mug"]

        paths = {
            "assets": os.path.join(loaded_path, "assets"),
            "bddl_files": os.path.join(loaded_path, "bddl_files"),
            "init_states": os.path.join(loaded_path, "init_files"),
        }
        os.environ["LIBERO_ASSET_ROOT"] = paths["assets"]
        os.environ["LIBERO_BDDL_PATH"] = paths["bddl_files"]
        os.environ["LIBERO_INIT_STATES_PATH"] = paths["init_states"]

        def force_local_path(path_name):
            return paths.get(path_name, os.path.join(loaded_path, path_name))
        libero_main.get_libero_path = force_local_path

        try:
            bddl_utils = importlib.import_module(f"{target_module_path}.envs.bddl_utils")
            
            _orig_get_info = bddl_utils.get_problem_info
            def safe_get_problem_info(bddl_file_path):
                try:
                    res = _orig_get_info(bddl_file_path)
                    if res.get("problem_name") != "unknown": return res
                except: pass
                p_name = "unknown"
                try:
                    with open(bddl_file_path, "r") as f:
                        content = f.read(2048)
                        import re
                        m = re.search(r'\(problem\s+([^\s\)]+)\)', content)
                        if m: p_name = m.group(1).lower()
                except: pass
                if p_name == "unknown": p_name = "libero_tabletop_manipulation"
                return {"domain_name": "robosuite", "problem_name": p_name, "language_instruction": "task"}
            
            bddl_utils.get_problem_info = safe_get_problem_info

            _orig_parse = bddl_utils.robosuite_parse_problem
            def safe_robosuite_parse_problem(bddl_file_path):
                try:
                    return _orig_parse(bddl_file_path)
                except Exception as e:
                    print(f"[Patch] Variant parse failed ({e}), falling back to original structure for: {os.path.basename(bddl_file_path)}")
                    
                    bddl_root = os.environ.get("LIBERO_BDDL_PATH", "")
                    
                    try:
                        suite_dir = os.path.join(bddl_root, "libero_10")
                        if not os.path.exists(suite_dir):
                            suite_dir = os.path.join(bddl_root, os.listdir(bddl_root)[0])
                        
                        fallback_file = os.path.join(suite_dir, os.listdir(suite_dir)[0])
                        return _orig_parse(fallback_file)
                    except:
                        raise e

            bddl_utils.robosuite_parse_problem = safe_robosuite_parse_problem
                        
        except (ImportError, AttributeError):
            pass  
    except Exception as e:
        print(f"[LiberoEnv Patch] Patching Error (pid={os.getpid()}): {e}")

apply_global_patches()

from rlinf.envs.libero.utils import (
    get_benchmark_overridden,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
)
from rlinf.envs.libero.venv import ReconfigureSubprocEnv
from rlinf.envs.utils import list_of_dict_to_dict_of_list, to_tensor


class LiberoEnv(gym.Env):
    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self.seed_offset = seed_offset
        self.cfg = cfg
        self.total_num_processes = total_num_processes
        self.worker_info = worker_info
        self.seed = self.cfg.seed + seed_offset
        self._is_start = True
        self.num_envs = num_envs
        self.group_size = self.cfg.group_size
        self.num_group = self.num_envs // self.group_size
        self.use_fixed_reset_state_ids = cfg.use_fixed_reset_state_ids
        self.specific_reset_id = cfg.get("specific_reset_id", None)

        self.ignore_terminations = cfg.ignore_terminations
        self.auto_reset = cfg.auto_reset

        self._generator = np.random.default_rng(seed=self.seed)
        self._generator_ordered = np.random.default_rng(seed=0)
        self.start_idx = 0

        self.task_descriptions = [""] * self.num_envs

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
                import importlib, os, sys
                os.environ["LIBERO_TYPE"] = _type_val
                
                apply_global_patches()
                
                if _type_val in ["pro", "plus"]:
                    try:
                        if "libero" not in sys.modules:
                            if _type_val == "pro": target_pkg, target_core = "liberopro", "liberopro.liberopro"
                            else: target_pkg, target_core = "liberoplus", "liberoplus.liberoplus"
                            real_libero_pkg = importlib.import_module(target_pkg)
                            real_libero_core = importlib.import_module(target_core)
                            try: real_libero_bench = importlib.import_module(f"{target_core}.benchmark")
                            except: real_libero_bench = importlib.import_module(f"{target_pkg}.benchmark")
                            real_libero_envs = importlib.import_module(f"{target_core}.envs")
                            sys.modules["libero"] = real_libero_pkg
                            sys.modules["libero.libero"] = real_libero_core
                            sys.modules["libero.libero.benchmark"] = real_libero_bench
                            sys.modules["libero.libero.envs"] = real_libero_envs
                    except ImportError: pass
                # -------------------------------------------------

                seed = param.pop("seed")
                
                try:
                    if _type_val == "pro": from liberopro.liberopro.envs import OffScreenRenderEnv as WorkerEnv
                    elif _type_val == "plus": from liberoplus.liberoplus.envs import OffScreenRenderEnv as WorkerEnv
                    else: from libero.libero.envs import OffScreenRenderEnv as WorkerEnv
                except ImportError:
                    import libero.libero.envs as _le
                    WorkerEnv = _le.OffScreenRenderEnv

                if _type_val == "plus":
                    try:
                        ParentEnv = WorkerEnv.__bases__[0]
                        if not getattr(ParentEnv, "_is_patched_by_rlinf", False):
                            orig_init = ParentEnv.__init__
                            def patched_init(self, **kwargs):
                                if "bddl_file_name" not in kwargs and "bddl_file_name" in param:
                                    kwargs["bddl_file_name"] = param["bddl_file_name"]
                                return orig_init(self, **kwargs)
                            ParentEnv.__init__ = patched_init
                            ParentEnv._is_patched_by_rlinf = True
                    except Exception: pass

                env = WorkerEnv(**param)
                env.seed(seed)
                return env

            env_fns.append(env_fn)
        return env_fns
    

    def get_env_fn_params(self, env_idx=None):
        env_fn_params = []
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        
        variant = os.environ.get("LIBERO_TYPE", self.cfg.get("libero_variant", "standard"))
        raw_suffix = os.environ.get("LIBERO_SUFFIX", self.cfg.get("perturbation_suffix", None))
        
        pro_suffix = raw_suffix.replace(".bddl", "") if (variant == "pro" and raw_suffix) else None
        
        bddl_root = get_libero_path("bddl_files")
        
        if env_idx is None:
            env_idx = np.arange(self.num_envs)
        
        suite_name = self.cfg.task_suite_name.lower()
        suite_keyword = suite_name.replace("libero_", "").strip() 
        
        task_descriptions = []
        

        for i, env_id in enumerate(env_idx):
            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder 
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)
            
            final_path = None
            
            # --- Pro ---
            if variant == "pro":
                if pro_suffix == "all":
                    all_sub_dirs = [
                        d for d in os.listdir(bddl_root) 
                        if os.path.isdir(os.path.join(bddl_root, d)) and suite_keyword in d
                    ]
                    
                    core_task_name = file_name.replace(".bddl", "")
                    all_candidates = []
                    
                    for sub_dir in all_sub_dirs:
                        target_dir_path = os.path.join(bddl_root, sub_dir)
                        matches = [
                            os.path.join(target_dir_path, f) 
                            for f in os.listdir(target_dir_path) 
                            if core_task_name in f and f.endswith(".bddl")
                        ]
                        all_candidates.extend(matches)
                    
                    if all_candidates:
                        all_candidates.sort()
                        final_path = all_candidates[(self.seed + i) % len(all_candidates)]
                    else:
                        final_path = original_path
                else:
                    final_path = original_path

            # --- Plus  ---
            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None
                if plus_suffix == "all":
                    clean_name = file_name.replace(".bddl", "")
                    for marker in ['_view', '_initstate', '_noise', '_sample', '_light', '_table', '_add', '_lan', '_language', '_copy', '_level', '_tb']:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break
                    
                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [d for d in os.listdir(bddl_root) if os.path.isdir(os.path.join(bddl_root, d))]
                    search_dirs = [os.path.join(bddl_root, d) for d in all_dirs if suite_pattern in d.lower().replace("_", "")]
                    
                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]
                    
                    all_candidates = []
                    for target_dir in search_dirs:
                        import glob
                        matches = [f for f in glob.glob(os.path.join(target_dir, "*.bddl")) if clean_name in os.path.basename(f)]
                        all_candidates.extend(matches)
                    
                    if all_candidates:
                        all_candidates.sort()
                        final_path = all_candidates[(self.seed + i) % len(all_candidates)]

            if final_path is None:
                final_path = original_path
                
            env_fn_params.append({
                **base_env_args,
                "bddl_file_name": final_path,
                "seed": self.seed,
            })
            task_descriptions.append(task.language)
            
        self.task_descriptions = task_descriptions
        if len(env_fn_params) != len(env_idx):
             print(f"CRITICAL WARNING: Length mismatch in get_env_fn_params! Expected {len(env_idx)}, Got {len(env_fn_params)}")
             
        return env_fn_params
    

    def get_env_fn_params(self, env_idx=None):
        base_env_args = OmegaConf.to_container(self.cfg.init_params, resolve=True)
        variant = os.environ.get("LIBERO_TYPE", self.cfg.get("libero_variant", "standard"))
        raw_suffix = os.environ.get("LIBERO_SUFFIX", self.cfg.get("perturbation_suffix", "all"))
        pro_suffix = raw_suffix.replace(".bddl", "") if (variant == "pro" and raw_suffix) else None
        
        bddl_root = get_libero_path("bddl_files")

        all_descriptions = []
        for i in range(self.num_envs):
            try:
                t = self.task_suite.get_task(self.task_ids[i])
                all_descriptions.append(t.language)
            except:
                all_descriptions.append("unknown task")
        self.task_descriptions = all_descriptions

        active_idx = env_idx if env_idx is not None else np.arange(self.num_envs)
        env_fn_params = []
        
        suite_keyword = self.cfg.task_suite_name.replace("libero_", "").strip()

        for i, env_id in enumerate(active_idx):
            task = self.task_suite.get_task(self.task_ids[env_id])
            folder_name = task.problem_folder  # "libero_10"
            file_name = task.bddl_file
            original_path = os.path.join(bddl_root, folder_name, file_name)
            final_path = None

            # --- Pro  ---
            if variant == "pro":
                variant_tags = ["object", "swap", "lan", "task"]
                core_task_name = file_name.replace(".bddl", "")
                all_candidates = []

                for d in os.listdir(bddl_root):
                    if suite_keyword in d and any(tag in d.lower() for tag in variant_tags):
                        if d != folder_name:
                            target_dir = os.path.join(bddl_root, d)
                            if os.path.isdir(target_dir):
                                for f in os.listdir(target_dir):
                                    if core_task_name in f and f.endswith(".bddl"):
                                        all_candidates.append(os.path.join(target_dir, f))
                
                if all_candidates:
                    all_candidates.sort()
                    final_path = all_candidates[(self.seed + env_id) % len(all_candidates)]
                    # print(f"[SUCCESS] Env {env_id} switched to PRO variant: {os.path.basename(final_path)}")
                else:
                    print(f"[WARNING] Env {env_id} could not find any PRO variants for {core_task_name}")

            # --- Plus  ---
            elif variant == "plus":
                plus_suffix = raw_suffix.replace(".bddl", "") if raw_suffix else None
                if plus_suffix == "all":
                    clean_name = file_name.replace(".bddl", "")
                    for marker in ['_view', '_initstate', '_noise', '_sample', '_light', '_table', '_add', '_lan', '_language', '_copy', '_level', '_tb']:
                        if marker in clean_name:
                            clean_name = clean_name.split(marker)[0]
                            break
                    suite_pattern = folder_name.replace("_", "").lower()
                    all_dirs = [d for d in os.listdir(bddl_root) if os.path.isdir(os.path.join(bddl_root, d))]
                    search_dirs = [os.path.join(bddl_root, d) for d in all_dirs if suite_pattern in d.lower().replace("_", "")]
                    if not search_dirs:
                        search_dirs = [os.path.join(bddl_root, folder_name)]
                    all_candidates = []

                    for target_dir in search_dirs:
                        import glob
                        matches = [f for f in glob.glob(os.path.join(target_dir, "*.bddl")) if clean_name in os.path.basename(f)]
                        all_candidates.extend(matches)
                    if all_candidates:
                        all_candidates.sort()
                        final_path = all_candidates[(self.seed + env_id) % len(all_candidates)]

            if final_path is None:
                final_path = original_path

            env_fn_params.append({
                **base_env_args, 
                "bddl_file_name": final_path, 
                "seed": self.seed
            })

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
        if self.cfg.is_eval or self.cfg.use_ordered_reset_state_ids:
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

    def _wrap_obs(self, obs_list):
        images_and_states_list = []
        for obs in obs_list:
            images_and_states = self._extract_image_and_state(obs)
            images_and_states_list.append(images_and_states)

        images_and_states = to_tensor(
            list_of_dict_to_dict_of_list(images_and_states_list)
        )

        full_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["full_image"]]
        )
        wrist_image_tensor = torch.stack(
            [value.clone() for value in images_and_states["wrist_image"]]
        )

        states = images_and_states["state"]

        obs = {
            "main_images": full_image_tensor,
            "wrist_images": wrist_image_tensor,
            "states": states,
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
        if variant not in ["plus", "pro"]:
            init_state = self._get_reset_states(env_idx=env_idx)
            self.env.set_init_state(init_state=init_state, id=env_idx)

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        reset_state_ids=None,
    ):
        if env_idx is None:
            env_idx = np.arange(self.num_envs)

        if self.is_start:
            reset_state_ids = (
                self.reset_state_ids if self.use_fixed_reset_state_ids else None
            )
            self._is_start = False

        if reset_state_ids is None:
            num_reset_states = len(env_idx)
            reset_state_ids = self._get_random_reset_state_ids(num_reset_states)

        self._reconfigure(reset_state_ids, env_idx)
        for _ in range(15):
            zero_actions = np.zeros((len(env_idx), 7))
            if self.cfg.reset_gripper_open:
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
        """Step the environment with the given actions."""
        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        self._elapsed_steps += 1
        raw_obs, _reward, terminations, info_lists = self.env.step(actions)
        self.current_raw_obs = raw_obs
        infos = list_of_dict_to_dict_of_list(info_lists)
        truncations = self.elapsed_steps >= self.cfg.max_episode_steps
        obs = self._wrap_obs(raw_obs)

        step_reward = self._calc_step_reward(terminations)

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
        # chunk_actions: [num_envs, chunk_step, action_dim]
        chunk_size = chunk_actions.shape[1]
        obs_list = []
        infos_list = []

        chunk_rewards = []

        raw_chunk_terminations = []
        raw_chunk_truncations = []
        for i in range(chunk_size):
            actions = chunk_actions[:, i]
            extracted_obs, step_reward, terminations, truncations, infos = self.step(
                actions, auto_reset=False
            )
            obs_list.append(extracted_obs)
            infos_list.append(infos)

            chunk_rewards.append(step_reward)
            raw_chunk_terminations.append(terminations)
            raw_chunk_truncations.append(truncations)

        chunk_rewards = torch.stack(chunk_rewards, dim=1)  # [num_envs, chunk_steps]
        raw_chunk_terminations = torch.stack(
            raw_chunk_terminations, dim=1
        )  # [num_envs, chunk_steps]
        raw_chunk_truncations = torch.stack(
            raw_chunk_truncations, dim=1
        )  # [num_envs, chunk_steps]

        past_terminations = raw_chunk_terminations.any(dim=1)
        past_truncations = raw_chunk_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
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
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(self, dones, _final_obs, infos):
        final_obs = copy.deepcopy(_final_obs)
        env_idx = np.arange(0, self.num_envs)[dones]
        final_info = copy.deepcopy(infos)
        if self.cfg.is_eval:
            self.update_reset_state_ids()
        obs, infos = self.reset(
            env_idx=env_idx,
            reset_state_ids=self.reset_state_ids[env_idx]
            if self.use_fixed_reset_state_ids
            else None,
        )
        # gymnasium calls it final observation but it really is just o_{t+1} or the true next observation
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
            img = raw_single_obs["agentview_image"][::-1, ::-1]
            img = put_info_on_image(img, info_item)
            images.append(img)
        full_image = tile_images(images, nrows=int(np.sqrt(self.num_envs)))
        self.render_images.append(full_image)

    def flush_video(self, video_sub_dir: Optional[str] = None):
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

