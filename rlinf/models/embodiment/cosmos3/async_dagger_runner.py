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

from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner


class Cosmos3AsyncDAggerRunner(AsyncEmbodiedRunner):
    """Use runner steps as rollout versions while Cosmos tracks optimizer steps."""

    def update_rollout_weights(self, no_wait: bool = False) -> None:
        self.actor.set_global_step(self.global_step).wait()
        super().update_rollout_weights(no_wait=no_wait)

    def run(self) -> None:
        # Restore the rollout policy's step before its first weight load.
        self.rollout.set_global_step(self.global_step).wait()
        super().run()
        self._finish_run()
