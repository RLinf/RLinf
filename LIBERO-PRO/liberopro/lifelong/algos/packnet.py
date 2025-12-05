import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from liberopro.liberopro.benchmark import *
from libero.lifelong.algos.base import Sequential
from libero.lifelong.metric import *
from libero.lifelong.utils import *


class PackNet(Sequential):
    """
    The PackNet policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.cfg = cfg
        previous_masks = {}
        for module_idx, module in enumerate(self.policy.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = torch.ByteTensor(module.weight.data.size()).fill_(0)
                if "cuda" in module.weight.data.type():
                    mask = mask.to(self.cfg.device)
                previous_masks[module_idx] = mask
        self.previous_masks = previous_masks

    def pruning_mask(self, weights, previous_mask, layer_idx):
        """
        Ranks weights by magnitude. Sets all below kth to 0.
        Returns pruned mask.
        """
        # Select all prunable weights, ie. belonging to current dataset.
        previous_mask = previous_mask.to(self.cfg.device)
        tensor = weights[
            previous_mask.eq(self.current_task + 1)
        ]  # current_task starts from 0, so we add 1
        abs_tensor = tensor.abs()
        cutoff_rank = round(self.cfg.lifelong.prune_perc * tensor.numel())
        cutoff_value = abs_tensor.view(-1).cpu().kthvalue(cutoff_rank)[0]

        # Remove those weights which are below cutoff and belong to current
        # dataset that we are training for.
        remove_mask = weights.abs().le(cutoff_value) * previous_mask.eq(
            self.current_task + 1
        )

        # mask is 1 - remove_mask
        previous_mask[remove_mask.eq(1)] = 0
        mask = previous_mask
        print(
            "Layer #%d, pruned %d/%d (%.2f%%) (Total in layer: %d)"
            % (
                layer_idx,
                mask.eq(0).sum(),
                tensor.numel(),
                100 * mask.eq(0).sum() / tensor.numel(),
                weights.numel(),
            )
        )
        return mask

    def prune(self):
        """
        Gets pruning mask for each layer, based on previous_masks.
        Sets the self.current_masks to the computed pruning masks.
        """
        self.current_masks = {}
        print(
            "[info] pruning each layer by removing %.2f%% of values"
            % (100 * self.cfg.lifelong.prune_perc)
        )

        for module_idx, module in enumerate(self.policy.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.pruning_mask(
                    module.weight.data, self.previous_masks[module_idx], module_idx
                )
                self.current_masks[module_idx] = mask.to(self.cfg.device)

                # Set pruned weights to 0.
                weight = module.weight.data
                weight[self.current_masks[module_idx].eq(0)] = 0.0
        self.previous_masks = self.current_masks

    def make_grads_zero(self):
        """
        Sets grads of fixed weights and Norm layers to 0.
        """
        assert self.current_masks

        for module_idx, module in enumerate(self.policy.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                layer_mask = self.current_masks[module_idx]

                # Set grads of all weights not belonging to current dataset to 0.
                if module.weight.grad is not None:
                    module.weight.grad.data[layer_mask.ne(self.current_task + 1)] = 0
                    # Biases are fixed.
                    if module.bias is not None:
                        module.bias.grad.data.fill_(0)
            elif "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                # Set grads of batchnorm params to 0.
                module.weight.grad.data.fill_(0)
                module.bias.grad.data.fill_(0)

    def start_task(self, task):
        """
        Turns previously pruned weights into trainable weights for current dataset.
        """
        super().start_task(task)
        assert self.previous_masks
        for module_idx, module in enumerate(self.policy.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                mask = self.previous_masks[module_idx]
                # since current_task starts from 0, we add 1
                mask[mask.eq(0)] = self.current_task + 1
            # we never train norm layers
            elif "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()

        self.current_masks = self.previous_masks

    def observe(self, data):
        # make norm layer to eval
        for module_idx, module in enumerate(self.policy.modules()):
            if "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()
        data = self.map_tensor_to_device(data)

        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (loss * self.loss_scale).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )

        # Set fixed param grads to 0.
        self.make_grads_zero()
        self.optimizer.step()
        return loss.item()

    def end_task(self, dataset, task_id, benchmark):
        # prune + post_finetune
        # for fair comparisons with other lifelong learning algorithms,
        # we do not use the success rates in the post_finetune epochs to AUC
        self.prune()

        # Do final finetuning to improve results on pruned network.
        if self.cfg.lifelong.post_prune_epochs:
            print("[info] start finetuning after pruning ...")
            # Note: here we do not apply start_task() to keep the 0 value in the
            # mask stay 0 and only update the param where mask=current_task+1
            # re-initialize the optimizer and scheduler
            self.optimizer = eval(self.cfg.train.optimizer.name)(
                self.policy.parameters(), **self.cfg.train.optimizer.kwargs
            )

            self.scheduler = None
            if self.cfg.train.scheduler is not None:
                self.scheduler = eval(self.cfg.train.scheduler.name)(
                    self.optimizer,
                    T_max=self.cfg.train.n_epochs,
                    **self.cfg.train.scheduler.kwargs,
                )

            model_checkpoint_name = os.path.join(
                self.experiment_dir, f"task{task_id}_model.pth"
            )

            train_dataloader = DataLoader(
                dataset,
                batch_size=self.cfg.train.batch_size,
                num_workers=self.cfg.train.num_workers,
                shuffle=True,
            )

            prev_success_rate = -1.0
            best_state_dict = self.policy.state_dict()  # currently save the best model
            torch_save_model(
                self.policy,
                model_checkpoint_name,
                cfg=self.cfg,
                previous_masks=self.previous_masks,
            )

            # this is just a fake summary object that works for placeholders
            sim_states = [[] for _ in range(self.cfg.eval.n_eval)]
            for epoch in range(0, self.cfg.lifelong.post_prune_epochs + 1):
                t0 = time.time()
                self.policy.train()
                training_loss = 0.0
                for (idx, data) in enumerate(train_dataloader):
                    loss = self.observe(data)
                    training_loss += loss
                training_loss /= len(train_dataloader)
                t1 = time.time()

                print(
                    f"[info] Post epoch: {epoch:3d} | train loss: {training_loss:5.2f} | time: {(t1-t0)/60:4.2f}"
                )
                time.sleep(0.1)

                if epoch % self.cfg.lifelong.post_eval_every == 0:  # evaluate BC loss
                    self.policy.eval()

                    t0 = time.time()
                    task = benchmark.get_task(task_id)
                    task_emb = benchmark.get_task_emb(task_id)
                    task_str = f"k{task_id}_e{epoch//self.cfg.lifelong.post_eval_every}"

                    success_rate = evaluate_one_task_success(
                        self.cfg,
                        self,
                        task,
                        task_emb,
                        task_id,
                        sim_states=sim_states,
                        task_str="",
                    )

                    if prev_success_rate < success_rate:
                        # we do not record the success rate
                        torch_save_model(
                            self.policy,
                            model_checkpoint_name,
                            cfg=self.cfg,
                            previous_masks=self.previous_masks,
                        )
                        prev_success_rate = success_rate

                    t1 = time.time()

                    ci = confidence_interval(success_rate, self.cfg.eval.n_eval)
                    print(
                        f"[info] Epoch: {epoch:3d} | succ: {success_rate:4.2f} ± {ci:4.2f}"
                        + f"best succ: {prev_success_rate} "
                        + f"| time: {(t1-t0)/60:4.2f}"
                    )

                if self.scheduler is not None:
                    self.scheduler.step()

            self.policy.load_state_dict(torch_load_model(model_checkpoint_name)[0])

    def get_eval_algo(self, task_id):
        # TODO: find a better way to do this
        # save and load a new model and set all params where mask > current_task + 1 to 0
        torch_save_model(
            self.policy,
            os.path.join(self.experiment_dir, "tmp_model.pth"),
            cfg=self.cfg,
        )
        eval_algo = safe_device(
            eval(self.cfg.lifelong.algo)(
                eval(self.cfg.benchmark_name)().n_tasks, self.cfg
            ),
            self.cfg.device,
        )
        model_state_dict, _, _ = torch_load_model(
            os.path.join(self.experiment_dir, "tmp_model.pth")
        )
        eval_algo.policy.load_state_dict(model_state_dict)

        eval_algo.previous_masks = self.previous_masks
        eval_algo.pruning_mask = self.pruning_mask
        eval_algo.current_masks = self.current_masks
        eval_algo.current_task = self.current_task
        eval_algo.optimizer = self.optimizer
        eval_algo.scheduler = self.scheduler
        eval_algo.experiment_dir = self.experiment_dir

        eval_algo.eval()

        for module_idx, module in enumerate(eval_algo.policy.modules()):
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                weight = module.weight.data
                mask = eval_algo.previous_masks[module_idx].to(self.cfg.device)
                weight[mask.eq(0)] = 0.0
                weight[mask.gt(task_id + 1)] = 0.0
            # we never train norm layers
            elif "BatchNorm" in str(type(module)) or "LayerNorm" in str(type(module)):
                module.eval()
        return eval_algo
