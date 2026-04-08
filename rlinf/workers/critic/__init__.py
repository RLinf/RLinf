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

from omegaconf import DictConfig

from rlinf.scheduler.worker.worker import Worker


def get_critic_worker(cfg: DictConfig) -> Worker:
    if cfg.critic.training_backend == "megatron":
        from .megatron_critic_worker import MegatronCritic

        return MegatronCritic
    else:
        raise ValueError(f"Unsupported training backend: {cfg.critic.training_backend}")
