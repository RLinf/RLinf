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


from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.profiler.profiler import ProfilerAction


# A simple context manager that does nothing, for disabled or waiting steps.
@contextmanager
def null_context():
    yield


class PyTorchProfiler:
    """
    A PyTorch Profiler wrapper that manages scheduling across multiple,
    non-continuous calls, making it suitable for use inside functions
    that are called repeatedly (like in an actor model).

    Usage (as intended by the user):
        profiler = PyTorchProfiler.from_config(cfg.profiler)

        # This function is called multiple times by an external loop
        def run_forward_backward():
            with profiler.step():
                # Your training code here
                ...
            profiler.advance_step()
    """

    def __init__(
        self,
        output_dir: str = "./profiler_output",
        activities: List[str] = ["cpu", "cuda"],
        schedule_wait: int = 5,
        schedule_warmup: int = 1,
        schedule_active: int = 3,
        schedule_repeat: int = 2,
        record_shapes: bool = False,
        profile_memory: bool = True,
        with_stack: bool = False,
        with_flops: bool = True,
        with_modules: bool = False,
        export_tensorboard: bool = True,
        export_chrome_trace: bool = True,
        chrome_filename_prefix: str = "chrome_trace",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.profiler_kwargs = {
            "activities": self._parse_activities(activities),
            "record_shapes": record_shapes,
            "profile_memory": profile_memory,
            "with_stack": with_stack,
            "with_flops": with_flops,
            "with_modules": with_modules,
        }

        self.on_trace_ready = self._create_trace_handler(
            export_tensorboard, export_chrome_trace, chrome_filename_prefix
        )

        self.schedule = schedule(
            wait=schedule_wait,
            warmup=schedule_warmup,
            active=schedule_active,
            repeat=schedule_repeat,
        )

        self.step_counter = 0
        self.current_action = ProfilerAction.NONE

    def _parse_activities(self, activity_strs: List[str]) -> List[ProfilerActivity]:
        valid_activities = set()
        activity_map = {"cpu": ProfilerActivity.CPU, "cuda": ProfilerActivity.CUDA}
        for act_str in activity_strs:
            act_enum = activity_map.get(act_str.lower())
            if act_enum:
                if act_enum == ProfilerActivity.CUDA and not torch.cuda.is_available():
                    print(
                        "Warning: 'cuda' activity requested but CUDA is not available."
                    )
                else:
                    valid_activities.add(act_enum)
            else:
                raise ValueError(f"Unknown profiler activity '{act_str}'.")
        return list(valid_activities)

    def _get_chrome_trace_filename(self, prefix: str) -> str:
        fname = prefix
        if torch.distributed.is_initialized():
            fname += f"_rank{torch.distributed.get_rank()}"
        return f"{fname}_{self.step_counter}.json"

    def _create_trace_handler(
        self, export_tb: bool, export_chrome: bool, chrome_prefix: str
    ) -> Optional[Callable]:
        if not (export_tb or export_chrome):
            return None

        if export_tb:
            tb_dir = str(self.output_dir / "tensorboard")
            print(f"Profiler configured to save TensorBoard traces to: {tb_dir}")
            return tensorboard_trace_handler(dir_name=tb_dir)

        if export_chrome:

            def chrome_handler(p):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trace_path = self.output_dir / self._get_chrome_trace_filename(
                    chrome_prefix
                )
                print(f"Profiler saving Chrome trace to: {trace_path}")
                try:
                    p.export_chrome_trace(str(trace_path))
                except Exception as e:
                    print(f"Failed to export Chrome trace: {e}")

            return chrome_handler

        return None  # should not be reached

    @contextmanager
    def step(self) -> contextmanager:
        self.current_action = self.schedule(self.step_counter)

        if self.current_action != ProfilerAction.NONE:
            temp_profiler = profile(
                **self.profiler_kwargs,
                on_trace_ready=self.on_trace_ready
                if self.current_action == ProfilerAction.RECORD
                else None,
            )
            with temp_profiler:
                yield
        else:
            # in a 'wait' step, do nothing.
            yield

    def advance_step(self):
        """Advances the profiler's step counter."""
        self.step_counter += 1

    @classmethod
    def from_config(
        cls, profiler_config: Optional[Dict[str, Any]]
    ) -> "PyTorchProfiler":
        """
        Creates a PyTorchProfiler instance from a configuration dictionary.
        Returns a disabled profiler if config is None or disabled.
        """
        required_params = {
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
            "schedule_wait",
            "schedule_warmup",
            "schedule_active",
            "schedule_repeat",
        }

        unknown_params = set(profiler_config.keys()) - required_params
        if unknown_params:
            raise ValueError(f"Unknown profiler parameters: {unknown_params}")

        valid_config = {
            k: v for k, v in profiler_config.items() if k in required_params
        }

        missing_params = set(required_params) - set(valid_config.keys())
        if missing_params:
            raise ValueError(f"Missing required profiler parameters: {missing_params}")

        return cls(**valid_config)
