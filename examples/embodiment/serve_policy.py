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

"""Start a fixed-checkpoint policy service without a Ray cluster or robot."""

import hydra
from omegaconf import DictConfig

from rlinf.workers.rollout.grpc.policy_server import serve_policy


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="realworld_so101_policy_server",
)
def main(cfg: DictConfig) -> None:
    """Serve the policy configured by the YAML, until interrupted."""
    try:
        serve_policy(cfg)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
