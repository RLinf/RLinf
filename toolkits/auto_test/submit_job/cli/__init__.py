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

"""submit_job.cli 包兼容层。"""

from cli.cli import main
from cli.predict import predict_with_model
from cli.push import push_training_job
from cli.train_service_idle_resources import create_idle_resource_train_service
from cli.training import train_with_model

__all__ = [
    "main",
    "push_training_job",
    "predict_with_model",
    "train_with_model",
    "create_idle_resource_train_service",
]
