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

"""Opt-in parameter-update diagnostics for unsharded ART actors."""

import torch


class ARTParameterProbe:
    """Keep a CPU reference to trainable weights and FP32 master weights.

    One reference spans both ART updates. Diagnostics intentionally transfer
    parameters to CPU; enable them for short numerical checks, not long runs.
    """

    def __init__(self, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
        self.parameters = []
        for parameter in model.parameters():
            if not parameter.requires_grad:
                continue
            master = optimizer.state.get(parameter, {}).get("fp32_master_param")
            self.parameters.append(
                (
                    parameter,
                    parameter.detach().cpu().clone(),
                    master,
                    None if master is None else master.detach().cpu().clone(),
                )
            )

    @torch.no_grad()
    def metrics(self, prefix: str, device: torch.device) -> dict[str, float]:
        """Return cumulative change from the pre-u1 reference, by weight dtype."""
        categories = ("bf16", "fp16", "fp32", "master")
        stats = torch.zeros(len(categories), 3, dtype=torch.float64)
        for parameter, before, master, master_before in self.parameters:
            dtype = {
                torch.bfloat16: "bf16",
                torch.float16: "fp16",
                torch.float32: "fp32",
            }[parameter.dtype]
            pairs = [(dtype, parameter, before)]
            if master is not None:
                pairs.append(("master", master, master_before))
            for category, current, reference in pairs:
                current = current.detach().cpu()
                delta = current.float() - reference.float()
                index = categories.index(category)
                stats[index, 0] += current.numel()
                stats[index, 1] += (current != reference).sum()
                stats[index, 2] += delta.square().sum(dtype=torch.float64)
        stats = stats.to(device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats)
            # This probe is restricted to NO_SHARD: each rank owns a replica.
            stats /= torch.distributed.get_world_size()
        result = {}
        for category, (count, changed, squared) in zip(categories, stats.tolist()):
            if count:
                result[f"{prefix}_{category}_numel"] = count
                result[f"{prefix}_{category}_changed_frac"] = changed / count
                result[f"{prefix}_{category}_update_rms"] = (squared / count) ** 0.5
        return result
