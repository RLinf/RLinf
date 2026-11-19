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

"""Driver ownership of Teacher worker groups."""

from dataclasses import asdict, dataclass, field

from omegaconf import OmegaConf

from rlinf.algorithms.distillation.config import (
    DistillationLossConfig,
    TeacherModelConfig,
)
from rlinf.utils.logging import get_logger

from .client import TeacherEndpoint
from .teacher_worker import TeacherWorker


@dataclass
class TeacherRuntime:
    """Driver-only handles; remote workers receive only endpoints."""

    worker_groups: list = field(default_factory=list)
    endpoints: list[TeacherEndpoint] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)

    def take_metrics(self) -> dict[str, float]:
        """Sum service counters for each model over its replicas."""
        metrics = {}
        handles = [group.take_metrics() for group in self.worker_groups]
        for endpoint, handle in zip(self.endpoints, handles, strict=True):
            for name, value in handle.wait()[0].items():
                key = f"teacher/{endpoint.teacher_key}/{name}"
                metrics[key] = metrics.get(key, 0) + value
        return metrics

    def close(self) -> None:
        """Attempt all cleanup operations without masking a training exception."""
        groups, self.worker_groups = self.worker_groups, []
        self.endpoints = []
        for group in reversed(groups):
            try:
                group.stop().wait()
            except Exception:
                get_logger().exception("Failed to stop a Teacher worker group")


def launch_teacher_workers(cfg, cluster, placement) -> TeacherRuntime:
    """Launch frozen replicas, rolling back partially initialized groups."""
    runtime = TeacherRuntime()
    loss = DistillationLossConfig(**cfg.distillation.get("distillation_loss", {}))
    runtime.manifest = {
        "route_field": cfg.distillation.get("teacher_key", "data_source"),
        "loss": asdict(loss),
        "models": {},
    }
    try:
        for index, key in enumerate(sorted(cfg.distillation.teacher_models)):
            values = cfg.distillation.teacher_models[key]
            model = TeacherModelConfig(**values)
            model_state = {
                "index": index,
                "config": OmegaConf.to_container(values, resolve=True),
                "replicas": placement.get_replicas(key),
            }
            runtime.manifest["models"][key] = model_state
            for replica_id in range(model.num_replicas):
                group = TeacherWorker.create_group(key, model, loss).launch(
                    cluster,
                    name=f"teacher_{key}_{replica_id}",
                    placement_strategy=placement.get_strategy(key, replica_id),
                )
                runtime.worker_groups.append(group)
                endpoints = group.init_worker().wait()
                if len(endpoints) != 1 or endpoints[0].teacher_key != key:
                    raise ValueError(
                        f"Expected one ready endpoint for {key} replica {replica_id}"
                    )
                info = endpoints[0].model_info
                if model_state.setdefault("identity", asdict(info)) != asdict(info):
                    raise ValueError(
                        f"Replicas of Teacher {key} disagree on model identity"
                    )
                if runtime.endpoints:
                    first = runtime.endpoints[0].model_info
                    if (first.tokenizer_fingerprint, first.vocab_size) != (
                        info.tokenizer_fingerprint,
                        info.vocab_size,
                    ):
                        raise ValueError(
                            f"Teacher {key} has incompatible token IDs or output vocabulary"
                        )
                runtime.endpoints.extend(endpoints)
        return runtime
    except BaseException:
        runtime.close()
        raise
