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

"""Control messages exchanged with a patched SGLang scheduler.

SGLang 0.5.19 sends scheduler IPC as msgpack and only accepts
``msgspec.Struct`` payloads (``BaseReq``). Plain dataclasses raise
``TypeError: Cannot serialize object of type ... over msgpack IPC``.
Opaque fields (worker address, placement, OmegaConf, method results)
stay pickled inside a ``PickleWrapper``.
"""

from typing import Any, Optional

from sglang.srt.managers.io_struct import (
    BaseReq,
    PickleWrapper,
    hook_custom_types,
    unwrap_from_pickle,
    wrap_as_pickle,
)


def _seal(value: Any) -> Any:
    if value is None or isinstance(value, PickleWrapper):
        return value
    return wrap_as_pickle(value)


class AbortGenerationInput(BaseReq, kw_only=True):
    pass


class AbortGenerationOutput(BaseReq, kw_only=True):
    waiting_reqs: Optional[PickleWrapper] = None
    running_reqs: Optional[PickleWrapper] = None

    def __post_init__(self):
        self.waiting_reqs = _seal(self.waiting_reqs)
        self.running_reqs = _seal(self.running_reqs)


class TaskMethodInput(BaseReq, kw_only=True):
    method_name: str
    args: Optional[PickleWrapper] = None
    kwargs: Optional[PickleWrapper] = None

    def __post_init__(self):
        self.args = _seal(self.args)
        self.kwargs = _seal(self.kwargs)

    def unpacked_args(self) -> tuple[list, dict]:
        args = unwrap_from_pickle(self.args) or []
        kwargs = unwrap_from_pickle(self.kwargs) or {}
        return list(args), dict(kwargs)


class TaskMethodOutput(BaseReq, kw_only=True):
    method_name: str
    result: Optional[PickleWrapper] = None

    def __post_init__(self):
        self.result = _seal(self.result)

    def unpacked_result(self) -> Any:
        return unwrap_from_pickle(self.result)


class SyncHFWeightInput(BaseReq, kw_only=True):
    # The rollout's current pause mode during async training, None when not
    # paused (i.e. the existing synchronous training path). Same values as the
    # `rollout.pause_mode` config: "in_place" / "retract" / "drain".
    pause_mode: Optional[str] = None


class SyncHFWeightOutput(BaseReq, kw_only=True):
    pass


hook_custom_types(
    AbortGenerationInput,
    AbortGenerationOutput,
    TaskMethodInput,
    TaskMethodOutput,
    SyncHFWeightInput,
    SyncHFWeightOutput,
)
