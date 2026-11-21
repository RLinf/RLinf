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

"""Tianji Marvin Pro environment and LoopInfra bridge.

The package deliberately keeps the LoopInfra import optional.  Dummy and
playback tests can therefore run without a vendor SDK, cameras, or a GPU.
"""

from .tianji_backend import (
    DummyTianjiBackend,
    LoopInfraPolicyBackend,
    TianjiObservation,
)

__all__ = [
    "DummyTianjiBackend",
    "LoopInfraPolicyBackend",
    "TianjiConfig",
    "TianjiEnv",
    "TianjiObservation",
]


def __getattr__(name):
    # Stage1 is deliberately usable in the LoopInfra CPU environment, which
    # does not install Gymnasium.  Defer the Gym import until a caller asks
    # for the environment itself.
    if name in {"TianjiConfig", "TianjiEnv"}:
        from .tianji_env import TianjiConfig, TianjiEnv

        return {"TianjiConfig": TianjiConfig, "TianjiEnv": TianjiEnv}[name]
    raise AttributeError(name)
