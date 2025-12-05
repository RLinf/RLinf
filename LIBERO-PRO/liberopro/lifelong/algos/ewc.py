import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from libero.lifelong.algos.base import Sequential
from libero.lifelong.utils import *


class EWC(Sequential):
    """
    The Elastic Weight Consolidation policy.
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)
        self.checkpoint = None
        self.fish = None

    def get_params(self):
        return torch.cat([p.reshape(-1) for p in self.policy.parameters()])

    def get_grads(self):
        return torch.cat(
            [
                p.grad.reshape(-1)
                if p.grad is not None
                else torch.zeros_like(p).reshape(-1)
                for p in self.policy.parameters()
            ]
        )

    def penalty(self):
        if self.checkpoint is None:
            return safe_device(torch.tensor(0.0), self.cfg.device)
        else:
            penalty = (self.fish * ((self.get_params() - self.checkpoint) ** 2)).sum()
            return penalty

    def end_task(self, dataset, task_id, benchmark):
        self.policy.train()
        fish = torch.zeros_like(self.get_params())

        dataloader = DataLoader(
            dataset,
            batch_size=self.cfg.train.batch_size,
            shuffle=True,
            num_workers=self.cfg.train.num_workers,
        )

        for data in dataloader:
            data = TensorUtils.map_tensor(
                data, lambda x: safe_device(x, device=self.cfg.device)
            )
            self.policy.zero_grad()
            nll = self.policy.compute_loss(data, reduction="none")
            (-nll).mean().backward()
            grads = self.get_grads()
            fish += grads**2

        fish /= len(dataloader)

        if self.fish is None:
            self.fish = fish
        else:
            self.fish *= self.cfg.lifelong.gamma
            self.fish += fish

        self.checkpoint = self.get_params().data.clone()

    def observe(self, data):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        forward_loss = loss.item()
        if self.current_task > 0:
            loss += self.cfg.lifelong.e_lambda * self.penalty()
        assert not torch.isnan(loss)
        (loss * self.loss_scale).backward()
        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return forward_loss
