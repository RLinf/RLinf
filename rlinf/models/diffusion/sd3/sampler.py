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

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from diffusers.utils.torch_utils import randn_tensor

EULER_SOLVERS = {"sde", "ode"}
DPM_SOLVER_ORDERS = {"dpm1": 1, "dpm2": 2}


def _zero_logprob(latents: torch.Tensor) -> torch.Tensor:
    return torch.zeros(
        latents.shape[0],
        device=latents.device,
        dtype=latents.dtype,
    )


def sde_step_with_logprob(
    scheduler,
    model_output: torch.Tensor,
    timestep: torch.Tensor,
    sample: torch.Tensor,
    *,
    noise_level: float,
    prev_sample: torch.Tensor | None = None,
    compute_logprob: bool = True,
):
    model_output = model_output.float()
    sample = sample.float()
    prev_sample = None if prev_sample is None else prev_sample.float()

    scheduler_timesteps = scheduler.timesteps.to(
        device=timestep.device,
        dtype=torch.float32,
    )
    if scheduler_timesteps.numel() == 0:
        raise ValueError(
            "Scheduler timesteps are empty. Call retrieve_timesteps first."
        )

    timestep = timestep.reshape(-1).to(dtype=torch.float32)
    step_index = (timestep[:, None] - scheduler_timesteps[None, :]).abs().argmin(dim=1)
    prev_step_index = step_index + 1

    sigma = scheduler.sigmas[step_index].view(-1, *([1] * (sample.ndim - 1)))
    sigma_prev = scheduler.sigmas[prev_step_index].view(
        -1,
        *([1] * (sample.ndim - 1)),
    )
    sigma_max = scheduler.sigmas[1].item()
    dt = sigma_prev - sigma

    std = torch.sqrt(sigma / (1 - torch.where(sigma == 1, sigma_max, sigma)))
    std = std * noise_level
    mean = sample * (1 + std**2 / (2 * sigma) * dt)
    mean += model_output * (1 + std**2 * (1 - sigma) / (2 * sigma)) * dt

    diffusion_std = std * torch.sqrt(-dt)
    if prev_sample is None:
        noise = randn_tensor(
            model_output.shape,
            device=model_output.device,
            dtype=model_output.dtype,
        )
        prev_sample = mean + diffusion_std * noise

    if not compute_logprob:
        return prev_sample, None, mean, std

    log_prob = (
        -((prev_sample.detach() - mean) ** 2) / (2 * diffusion_std**2)
        - torch.log(diffusion_std)
        - torch.log(torch.sqrt(2 * torch.as_tensor(math.pi)))
    )
    log_prob = log_prob.mean(dim=tuple(range(1, log_prob.ndim)))
    return prev_sample, log_prob, mean, std


@dataclass
class _DPMState:
    order: int
    model_outputs: list[torch.Tensor | None] | None = None
    lower_order_nums: int = 0

    def __post_init__(self):
        self.model_outputs = [None] * self.order

    def update(self, model_output: torch.Tensor):
        for i in range(self.order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
        self.model_outputs[-1] = model_output

    def update_lower_order(self):
        if self.lower_order_nums < self.order:
            self.lower_order_nums += 1


def _sigma_to_alpha_sigma_t(sigma: torch.Tensor):
    return 1 - sigma, sigma


def _convert_to_x0_prediction(
    model_output: torch.Tensor,
    sample: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
) -> torch.Tensor:
    sigma_t = sigmas[step_index].to(device=sample.device, dtype=sample.dtype)
    return sample - sigma_t * model_output


def _deterministic_ddim_update(
    model_output: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    t, s = sigmas[step_index + 1], sigmas[step_index]
    noise_pred = (sample - (1 - s) * model_output) / s
    return (1 - t) * model_output + t * noise_pred


def _dpm_solver_first_order_update(
    model_output: torch.Tensor,
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_t, sigma_s = sigmas[step_index + 1], sigmas[step_index]
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s, sigma_s = _sigma_to_alpha_sigma_t(sigma_s)
    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s = torch.log(alpha_s) - torch.log(sigma_s)

    h = lambda_t - lambda_s
    return (sigma_t / sigma_s) * sample - (
        alpha_t * (torch.exp(-h) - 1.0)
    ) * model_output


def _dpm_solver_second_order_update(
    model_outputs: list[torch.Tensor | None],
    sigmas: torch.Tensor,
    step_index: int,
    sample: torch.Tensor,
) -> torch.Tensor:
    sigma_t, sigma_s0, sigma_s1 = (
        sigmas[step_index + 1],
        sigmas[step_index],
        sigmas[step_index - 1],
    )
    alpha_t, sigma_t = _sigma_to_alpha_sigma_t(sigma_t)
    alpha_s0, sigma_s0 = _sigma_to_alpha_sigma_t(sigma_s0)
    alpha_s1, sigma_s1 = _sigma_to_alpha_sigma_t(sigma_s1)

    lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
    lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)
    lambda_s1 = torch.log(alpha_s1) - torch.log(sigma_s1)

    m0, m1 = model_outputs[-1], model_outputs[-2]
    h, h_0 = lambda_t - lambda_s0, lambda_s0 - lambda_s1
    r0 = h_0 / h
    d0, d1 = m0, (1.0 / r0) * (m0 - m1)

    return (
        (sigma_t / sigma_s0) * sample
        - (alpha_t * (torch.exp(-h) - 1.0)) * d0
        - 0.5 * (alpha_t * (torch.exp(-h) - 1.0)) * d1
    )


def _dpm_step(
    *,
    order: int,
    model_output: torch.Tensor,
    sample: torch.Tensor,
    step_index: int,
    sigmas: torch.Tensor,
    dpm_state: _DPMState,
) -> torch.Tensor:
    num_timesteps = sigmas.shape[0] - 1
    lower_order_final = step_index == num_timesteps - 1
    lower_order_second = step_index == num_timesteps - 2 and num_timesteps < 15
    solver_sigmas = sigmas.to(device=sample.device, dtype=torch.float64)

    model_output = _convert_to_x0_prediction(
        model_output,
        sample,
        sigmas,
        step_index=step_index,
    )
    dpm_state.update(model_output)
    sample = sample.to(dtype=torch.float32)

    if order == 1 or dpm_state.lower_order_nums < 1 or lower_order_final:
        if step_index == 0 or lower_order_final:
            prev_sample = _deterministic_ddim_update(
                model_output,
                solver_sigmas,
                step_index,
                sample,
            )
        else:
            prev_sample = _dpm_solver_first_order_update(
                model_output,
                solver_sigmas,
                step_index,
                sample,
            )
    elif order == 2 or dpm_state.lower_order_nums < 2 or lower_order_second:
        prev_sample = _dpm_solver_second_order_update(
            dpm_state.model_outputs,
            solver_sigmas,
            step_index,
            sample,
        )
    else:
        raise NotImplementedError("Only DPM order 1 and 2 are supported.")

    dpm_state.update_lower_order()
    return prev_sample.to(dtype=model_output.dtype)


def init_solver_state(solver: str) -> tuple[str, _DPMState | None]:
    solver = solver.lower()
    if solver in EULER_SOLVERS:
        return solver, None
    if solver in DPM_SOLVER_ORDERS:
        return solver, _DPMState(order=DPM_SOLVER_ORDERS[solver])
    supported = sorted(EULER_SOLVERS | set(DPM_SOLVER_ORDERS))
    raise ValueError(
        f"Unsupported SD3 solver: {solver}. Supported solvers: {supported}"
    )


def model_timestep(
    t: torch.Tensor,
    *,
    sigmas: torch.Tensor,
    step_index: int,
    batch_size: int,
    use_dpm: bool,
) -> torch.Tensor:
    if not use_dpm:
        return t.expand(batch_size)
    return (sigmas[step_index] * 1000).expand(batch_size).to(dtype=torch.long)


def sample_latents_step(
    *,
    scheduler,
    solver: str,
    noise_pred: torch.Tensor,
    timestep: torch.Tensor,
    latents: torch.Tensor,
    noise_level: float,
    compute_logprob: bool,
    step_index: int,
    sigmas: torch.Tensor,
    dpm_state: _DPMState | None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if dpm_state is not None:
        latents = _dpm_step(
            order=dpm_state.order,
            model_output=noise_pred.float(),
            sample=latents.float(),
            step_index=step_index,
            sigmas=sigmas,
            dpm_state=dpm_state,
        )
        log_prob = _zero_logprob(latents) if compute_logprob else None
        return latents, log_prob

    use_sde_noise = solver == "sde" and noise_level > 0
    latents, log_prob, _, _ = sde_step_with_logprob(
        scheduler,
        noise_pred,
        timestep.unsqueeze(0),
        latents,
        noise_level=noise_level if use_sde_noise else 0.0,
        compute_logprob=compute_logprob and use_sde_noise,
    )
    if compute_logprob and log_prob is None:
        log_prob = _zero_logprob(latents)
    return latents, log_prob
