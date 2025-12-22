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


from pathlib import Path
from typing import Any, Callable, Optional

import torch
from torch.profiler import (
    ProfilerActivity,
    profile,
    schedule,
    tensorboard_trace_handler,
)
from torch.profiler.profiler import ProfilerAction


class PyTorchProfilerFunc:
    """
    Helper Class to create record_function and start/stop gracefully.
    """

    def __init__(self, name: str):
        self.context = torch.profiler.record_function(name)

    def start(self):
        self.context.__enter__()

    def stop(self):
        self.context.__exit__(None, None, None)


class PyTorchProfiler:
    def __init__(
        self,
        output_dir: str = "./profiler_output",
        activities: list[str] = ["cpu", "cuda"],
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

        self.on_fwd_trace_ready = self._create_trace_handler(
            export_tensorboard,
            export_chrome_trace,
            chrome_filename_prefix,
            forward_only=True,
        )
        self.on_fwd_bwd_trace_ready = self._create_trace_handler(
            export_tensorboard,
            export_chrome_trace,
            chrome_filename_prefix,
            forward_only=False,
        )

        self.fwd_bwd_step_counter = 0  # training's micro_batch steps
        self.fwd_step_counter = 0  # inference's micro_batch steps
        self.current_action = ProfilerAction.NONE  # schedule's status
        self.active_profiler = None  # current used profiler
        self.fwd_bwd_schedule = None  # schedule for training

        self.schedule_repeat = schedule_repeat
        self.schedule_active = schedule_active
        self.schedule_warmup = schedule_warmup
        self.schedule_wait = 0

    def _parse_activities(self, activity_strs: list[str]) -> list[ProfilerActivity]:
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
        return f"{fname}_{self.fwd_bwd_step_counter}.json"

    def _create_trace_handler(
        self,
        export_tb: bool,
        export_chrome: bool,
        chrome_prefix: str,
        forward_only: bool,
    ) -> Optional[Callable]:
        if not (export_tb or export_chrome):
            return None

        if export_tb:
            tb_dir = str(
                self.output_dir / "tensorboard" / ("fwd" if forward_only else "fwd_bwd")
            )
            return tensorboard_trace_handler(dir_name=tb_dir)

        if export_chrome:

            def chrome_handler(p):
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                trace_path = self.output_dir / self._get_chrome_trace_filename(
                    chrome_prefix
                )
                try:
                    p.export_chrome_trace(
                        str(trace_path) / ("fwd" if forward_only else "fwd_bwd")
                    )
                except Exception as e:
                    print(f"Failed to export Chrome trace: {e}")

            return chrome_handler

        return None  # should not be reached

    def init_fwd_bwd_schedule(self, num_minibatches: int = 0) -> None:
        if self.fwd_bwd_schedule is None:
            self.schedule_wait = (
                num_minibatches - self.schedule_warmup - self.schedule_active
            )
            assert self.schedule_wait >= 0, (
                f"schedule_wait should greater than or equal to 0,got {self.schedule_wait}"
            )
            self.fwd_bwd_schedule = schedule(
                wait=self.schedule_wait,
                warmup=self.schedule_warmup,
                active=self.schedule_active,
                repeat=self.schedule_repeat,
            )
        else:
            schedule_wait = (
                num_minibatches - self.schedule_warmup - self.schedule_active
            )
            assert schedule_wait == self.schedule_wait, (
                "num_minibatches changed, please create a new profiler instance,"
                f"old is {self.schedule_wait} new is {schedule_wait}"
            )

    def start(self, forward_only: bool = False) -> None:
        if self.active_profiler:
            raise RuntimeError(
                "Profiler is already running. Call stop() before start()."
            )

        if forward_only:
            self.active_profiler = profile(
                **self.profiler_kwargs,
                on_trace_ready=self.on_fwd_trace_ready
                if self.fwd_step_counter < self.schedule_repeat
                else None,
            )
            self.active_profiler.start()
        else:
            self.current_action = self.fwd_bwd_schedule(self.fwd_bwd_step_counter)
            if self.current_action != ProfilerAction.NONE:
                self.active_profiler = profile(
                    **self.profiler_kwargs,
                    on_trace_ready=self.on_fwd_bwd_trace_ready
                    if self.current_action == ProfilerAction.RECORD_AND_SAVE
                    else None,
                )
                self.active_profiler.start()

    def stop(self, forward_only: bool = False):
        if self.active_profiler:
            self.active_profiler.stop()
            self.active_profiler = None
        if forward_only:
            self.fwd_step_counter += 1
        else:
            self.fwd_bwd_step_counter += 1

    @classmethod
    def from_config(
        cls, profiler_config: Optional[dict[str, Any]]
    ) -> "PyTorchProfiler":
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
