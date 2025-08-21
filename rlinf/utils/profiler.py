# Copyright 2025 The RLinf Authors.
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

import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.profiler import ProfilerActivity, profile, tensorboard_trace_handler


# NEW: A simple context manager that does nothing.
@contextmanager
def null_context():
    yield


class PyTorchProfiler:
    """
    A PyTorch Profiler wrapper for RL training.

    This profiler is configured via a dictionary (e.g., from YAML) and used
    as a context manager that automatically handles whether to profile a given step.

    Usage:
        profiler = PyTorchProfiler.from_config(cfg.profiler)
        for step in range(num_steps):
            with profiler.step(step):
                # Your training code here
                ...
    """

    def __init__(
        self,
        output_dir: str = "./profiler_output",
        activities: List[str] = ["cpu", "cuda"],
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = False,
        export_tensorboard: bool = True,
        export_chrome_trace: bool = True,
        chrome_filename_prefix: str = "chrome_trace",
    ):
        self.step_idx = 0
        self.output_dir = Path(output_dir)
        self.activities = self._parse_activities(activities)

        self.on_trace_ready = self._create_trace_handler(
            export_tensorboard, export_chrome_trace, chrome_filename_prefix
        )
        self.profiler_instance = profile(
            activities=self.activities,
            record_shapes=record_shapes,
            profile_memory=profile_memory,
            with_stack=with_stack,
            with_flops=with_flops,
            with_modules=with_modules,
            on_trace_ready=self.on_trace_ready,
        )

        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _parse_activities(self, activity_strs: List[str]) -> List[ProfilerActivity]:
        """Parses string activities from config into ProfilerActivity enums."""
        valid_activities = set()
        activity_map = {
            "cpu": ProfilerActivity.CPU,
            "cuda": ProfilerActivity.CUDA,
        }
        for act_str in activity_strs:
            act_enum = activity_map.get(act_str.lower())
            if act_enum:
                if act_enum == ProfilerActivity.CUDA and not torch.cuda.is_available():
                    raise RuntimeError(
                        "'cuda' activity requested but CUDA is not available."
                    )
                else:
                    valid_activities.add(act_enum)
            else:
                raise ValueError(
                    f"Unknown profiler activity '{act_str}', currently support ['cpu', 'cuda']."
                )

        if ProfilerActivity.CUDA in valid_activities:
            valid_activities.add(ProfilerActivity.CPU)

        if not valid_activities:
            print(
                "Warning: No valid profiler activities enabled. Profiler will not run."
            )
            self.enabled = False

        return list(valid_activities)

    def _get_chrome_trace_filename(self, prefix: str) -> str:
        """Generates a unique filename for the Chrome trace."""
        fname = prefix
        if torch.distributed.is_initialized():
            fname += f"_rank{torch.distributed.get_rank()}"
        if torch.cuda.is_available():
            fname += f"_dev{torch.cuda.current_device()}"

        timestamp = int(time.time() * 1000)
        fname += f"_{timestamp}"
        return f"{fname}.json"

    def _create_trace_handler(
        self, export_tb: bool, export_chrome: bool, chrome_prefix: str
    ) -> Optional[Callable]:
        """Creates a composed handler for on_trace_ready."""
        if not (export_tb or export_chrome):
            return None

        handlers = []
        if export_tb:
            tb_dir = str(self.output_dir / "tensorboard")
            handlers.append(tensorboard_trace_handler(tb_dir, worker_name="worker"))

        if export_chrome:

            def chrome_handler(prof):
                trace_path = self.output_dir / self._get_chrome_trace_filename(
                    chrome_prefix
                )
                prof.export_chrome_trace(str(trace_path))
                print(f"Chrome trace exported to: {trace_path}")

            handlers.append(chrome_handler)

        def composed_handler(prof):
            # Synchronize before export for accurate timings
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            for handler in handlers:
                try:
                    handler(prof)
                except Exception as e:
                    print(f"Profiler trace handler '{handler.__name__}' error: {e}")

        return composed_handler

    @contextmanager
    def step(self) -> Optional[torch.profiler.profile]:
        with self.profiler_instance as prof:
            yield prof

    @classmethod
    def from_config(
        cls, profiler_config: Optional[Dict[str, Any]]
    ) -> "PyTorchProfiler":
        """
        Creates a PyTorchProfiler instance from a configuration dictionary.
        Returns a disabled profiler if config is None or disabled.
        """
        supported_params = {
            "output_dir",
            "activities",
            "record_shapes",
            "profile_memory",
            "with_stack",
            "with_flops",
            "with_modules",
            "export_tensorboard",
            "export_chrome_trace",
            "chrome_filename_prefix",
        }

        unknown_params = set(profiler_config.keys()) - supported_params
        if unknown_params:
            raise ValueError(f"Unknown profiler parameters: {unknown_params}")

        valid_config = {
            k: v for k, v in profiler_config.items() if k in supported_params
        }

        missing_params = set(supported_params) - set(valid_config.keys())
        if missing_params:
            raise ValueError(f"Missing required profiler parameters: {missing_params}")

        return cls(**valid_config)
