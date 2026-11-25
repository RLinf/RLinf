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

"""Cosmos3 sampling extension loaded by SGLANG_EXTERNAL_MODEL_PACKAGE."""

from sglang.multimodal_gen.configs.sample.cosmos3 import Cosmos3SamplingParams


class Cosmos3ActionSamplingParams(Cosmos3SamplingParams):
    """Preserve explicit action parameters when Edge fills variant defaults."""

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._explicit_fields = set(kwargs)


def register_cosmos3_action_sampling() -> None:
    """Register sampling config through SGLang's native pipeline extension API."""
    from sglang.multimodal_gen.configs.pipeline_configs.cosmos3 import Cosmos3Config
    from sglang.multimodal_gen.registry import register_pipeline
    from sglang.multimodal_gen.runtime.pipelines.cosmos3_pipeline import (
        Cosmos3OmniDiffusersPipeline,
        Cosmos3Pipeline,
    )

    for pipeline in (Cosmos3Pipeline, Cosmos3OmniDiffusersPipeline):
        register_pipeline(
            pipeline,
            sampling_param_cls=Cosmos3ActionSamplingParams,
            pipeline_config_cls=Cosmos3Config,
            overwrite=True,
        )


register_cosmos3_action_sampling()
