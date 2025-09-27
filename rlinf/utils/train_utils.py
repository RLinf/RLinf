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

from megatron.core.distributed import finalize_model_grads


def set_sync_funcs(megatron_model_manager, forward_only):
    # handle asynchronous grad reduction
    no_sync_func = None
    grad_sync_func = None
    param_sync_func = None
    if megatron_model_manager._cfg.optim.use_distributed_optimizer:
        if not forward_only:
            if megatron_model_manager._cfg.optim.get("overlap_grad_reduce", False):
                no_sync_func = [
                    model_chunk.no_sync for model_chunk in megatron_model_manager.model
                ]
                no_sync_func = (
                    no_sync_func[0]
                    if len(megatron_model_manager.model) == 1
                    else no_sync_func
                )

                if megatron_model_manager._cfg.optim.get("align_grad_reduce", True):
                    grad_sync_func = [
                        model_chunk.start_grad_sync
                        for model_chunk in megatron_model_manager.model
                    ]
                    grad_sync_func = (
                        grad_sync_func[0]
                        if len(megatron_model_manager.model) == 1
                        else grad_sync_func
                    )
            if megatron_model_manager._cfg.optim.get(
                "overlap_param_gather", False
            ) and megatron_model_manager._cfg.optim.get("align_param_gather", False):
                param_sync_func = [
                    model_chunk.start_param_sync
                    for model_chunk in megatron_model_manager.model
                ]
                param_sync_func = (
                    param_sync_func[0]
                    if len(megatron_model_manager.model) == 1
                    else param_sync_func
                )

    # pipeline schedules will get these from self.model.config
    for module in megatron_model_manager.get_model_module_list():
        module.config.no_sync_func = no_sync_func
        module.config.grad_sync_func = grad_sync_func
        module.config.param_sync_func = param_sync_func

        module.config.finalize_model_grads_func = finalize_model_grads
        if not forward_only:
            module.config.grad_scale_func = megatron_model_manager.optimizer.scale_loss
        else:
            module.config.grad_scale_func = None


def set_train(megatron_model_manager):
    if isinstance(megatron_model_manager.model, list):
        for model_module in megatron_model_manager.model:
            model_module.train()
    else:
        megatron_model_manager.model.train()


def set_eval(megatron_model_manager):
    if isinstance(megatron_model_manager.model, list):
        for model_module in megatron_model_manager.model:
            model_module.eval()
    else:
        megatron_model_manager.model.eval()
