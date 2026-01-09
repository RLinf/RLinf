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

import copy

import torch
import torch.distributed

from megatron.training.training import unwrap_model
from megatron.training.utils import average_losses_across_data_parallel_group
from omegaconf import DictConfig
from torch.multiprocessing.reductions import reduce_tensor

from rlinf.workers.megatron_worker import MegatronWorker
from rlinf.algorithms.losses import compute_ppo_critic_loss

try:
    from params_resharding import nccl_group_recreate

    HAVE_RESHARDING = True
except ImportError:
    HAVE_RESHARDING = False


from rlinf.utils.placement import ModelParallelComponentPlacement, PlacementMode
from rlinf.workers.megatron_worker import MegatronWorker

import sys

class MegatronCritic(MegatronWorker):
    def __init__(
        self, cfg: DictConfig, placement: ModelParallelComponentPlacement, role='critic'
    ):
        """Initialize the MegatronWorker.

        Args:
            cfg (DictConfig): The configuration for the actor.
        """
        super().__init__(cfg, placement, role)

        self.infer_forward_micro_batch_size = (
            self.cfg.algorithm.logprob_forward_micro_batch_size # check this
        )

    def get_forward_step_func(self):
        """Acquire the forward step function for the model."""

        def forward_output_and_loss_func(dataloader_iter, model):
            batch = next(dataloader_iter)

            batch = {key: val.cuda() for key, val in batch.items()}

            input_ids = batch["input_ids"]
            attention_mask = batch["attention_mask"]
            position_ids = batch["position_ids"]

            response_len = self.response_len
            responses = input_ids[:, -response_len:]
            label = copy.deepcopy(position_ids)
            label[:, -response_len - 1 : -1] = responses

            output = self.custom_forward(
                model,
                input_ids,
                attention_mask,
                position_ids,
                sequence_parallel=self.transformer_config.sequence_parallel,
                temperature=self.cfg.algorithm.sampling_params.temperature, # TODO check this
            )

            if not self.return_loss:

                def id_func(output, non_loss_data=True):
                    return output
                
                if unwrap_model(model).post_process:
                    mask = batch["response_mask"][:, -response_len:]
                    output = output[:, -response_len - 1 : -1].contiguous().squeeze(-1)
                    output = output * mask

                return output, id_func

            def loss_func(output):

                returns = batch["returns"]
                prev_values = batch["values"]
                vpreds = output[:, -response_len - 1 : -1].contiguous().squeeze(-1)

                mask = batch["response_mask"][:, -response_len:]

                loss, metrics_data = compute_ppo_critic_loss(values=vpreds,
                                                             returns=returns,
                                                             prev_values=prev_values,
                                                             value_clip=self.cfg.algorithm.value_cliprange,
                                                             huber_delta=10000,
                                                             loss_agg_func=self.loss_agg_func,
                                                             loss_mask=mask)

                metrics_data.update(
                    {
                        "critic/final_value_loss": loss.detach(),
                    }
                )

                for k, v in metrics_data.items():
                    if v is not None:
                        metrics_data[k] = average_losses_across_data_parallel_group([v])

                return loss, metrics_data

            return output, loss_func

        return forward_output_and_loss_func
