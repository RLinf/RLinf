import numpy as np
import robomimic.utils.tensor_utils as TensorUtils
import torch
import torch.nn as nn
import torch.nn.functional as F

from libero.lifelong.algos.er import ER
from libero.lifelong.utils import *


def project(gxy: torch.Tensor, ger: torch.Tensor) -> torch.Tensor:
    corr = torch.dot(gxy, ger) / torch.dot(ger, ger)
    return gxy - corr * ger


def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def store_grad(params, grads, grad_dims):
    """
    This stores parameter gradients of past tasks.
    pp: parameters
    grads: gradients
    grad_dims: list with number of parameters per layers
    """
    # store the gradients
    grads.fill_(0.0)
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = np.sum(grad_dims[: count + 1])
            grads[begin:end].copy_(param.grad.data.view(-1))
        count += 1


def overwrite_grad(params, newgrad, grad_dims):
    """
    This is used to overwrite the gradients with a new gradient
    vector, whenever violations occur.
    pp: parameters
    newgrad: corrected gradient
    grad_dims: list storing number of parameters at each layer
    """
    count = 0
    for param in params():
        if param.grad is not None:
            begin = 0 if count == 0 else sum(grad_dims[:count])
            end = sum(grad_dims[: count + 1])
            this_grad = newgrad[begin:end].contiguous().view(param.grad.data.size())
            param.grad.data.copy_(this_grad)
        count += 1


class AGEM(ER):
    """
    The Avaraged Gradient Episodic Memory algorithm.
    See https://openreview.net/forum?id=Hkf2_sC5FX
    """

    def __init__(self, n_tasks, cfg, **policy_kwargs):
        super().__init__(n_tasks=n_tasks, cfg=cfg, **policy_kwargs)

        self.grad_dims = []
        for pp in self.policy.parameters():
            self.grad_dims.append(pp.data.numel())
        self.grad_xy = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)
        self.grad_er = torch.Tensor(np.sum(self.grad_dims)).to(self.cfg.device)

    def observe(self, data):
        data = self.map_tensor_to_device(data)
        self.optimizer.zero_grad()
        loss = self.policy.compute_loss(data)
        (loss * self.loss_scale).backward()

        if self.buffer is not None:
            store_grad(self.policy.parameters, self.grad_xy, self.grad_dims)
            buf_data = next(self.buffer)
            self.policy.zero_grad()

            buf_data = self.map_tensor_to_device(buf_data)
            buf_loss = self.policy.compute_loss(buf_data)
            buf_loss.backward()
            store_grad(self.policy.parameters, self.grad_er, self.grad_dims)

            dot_prod = torch.dot(self.grad_xy, self.grad_er)
            if dot_prod.item() < 0:
                g_tilde = project(gxy=self.grad_xy, ger=self.grad_er)
                overwrite_grad(self.policy.parameters, g_tilde, self.grad_dims)
            else:
                overwrite_grad(self.policy.parameters, self.grad_xy, self.grad_dims)

        if self.cfg.train.grad_clip is not None:
            grad_norm = nn.utils.clip_grad_norm_(
                self.policy.parameters(), self.cfg.train.grad_clip
            )
        self.optimizer.step()
        return loss.item()
