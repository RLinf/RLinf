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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class TaskMethodInput:
    method_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskMethodOutput:
    method_name: str
    result: Optional[Any] = None


@dataclass
class OffloadReqInput:
    pass


@dataclass
class OffloadReqOutput:
    pass


@dataclass
class SyncWeightInput:
    pass


@dataclass
class SyncWeightOutput:
    pass


@dataclass
class SyncHFWeightInput:
    pass


@dataclass
class SyncHFWeightOutput:
    pass
