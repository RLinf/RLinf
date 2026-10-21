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

"""Simulated environments.

One subpackage per simulator or benchmark. Nothing is imported here: a node
needs only the simulator it actually runs, and several of these pull in heavy
optional dependencies. ``rlinf.envs.get_env_cls`` imports the one selected by
``env.train.env_type`` and nothing else.
"""
