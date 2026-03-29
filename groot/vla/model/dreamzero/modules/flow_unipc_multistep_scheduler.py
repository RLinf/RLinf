# Copied from https://github.com/huggingface/diffusers/blob/v0.31.0/src/diffusers/schedulers/scheduling_unipc_multistep.py
# Convert unipc for flow matching
# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.schedulers.scheduling_utils import (
    KarrasDiffusionSchedulers,
    SchedulerMixin,
    SchedulerOutput,
)


class FlowUniPCMultistepScheduler(SchedulerMixin, ConfigMixin):
    """
    `UniPCMultistepScheduler` is a training-free framework designed for the fast sampling of diffusion models.

    This model inherits from [`SchedulerMixin`] and [`ConfigMixin`]. Check the superclass documentation for the generic
    methods the library implements for all schedulers such as loading and saving.

    Args:
        num_train_timesteps (`int`, defaults to 1000):
            The number of diffusion steps to train the model.
        solver_order (`int`, default `2`):
            The UniPC order which can be any positive integer. The effective order of accuracy is `solver_order + 1`
            due to the UniC. It is recommended to use `solver_order=2` for guided sampling, and `solver_order=3` for
            unconditional sampling.
        prediction_type (`str`, defaults to "flow_prediction"):
            Prediction type of the scheduler function; must be `flow_prediction` for this scheduler, which predicts
            the flow of the diffusion process.
        thresholding (`bool`, defaults to `False`):
            Whether to use the "dynamic thresholding" method. This is unsuitable for latent-space diffusion models such
            as Stable Diffusion.
        dynamic_thresholding_ratio (`float`, defaults to 0.995):
            The ratio for the dynamic thresholding method. Valid only when `thresholding=True`.
        sample_max_value (`float`, defaults to 1.0):
            The threshold value for dynamic thresholding. Valid only when `thresholding=True` and `predict_x0=True`.
        predict_x0 (`bool`, defaults to `True`):
            Whether to use the updating algorithm on the predicted x0.
        solver_type (`str`, default `bh2`):
            Solver type for UniPC. It is recommended to use `bh1` for unconditional sampling when steps < 10, and `bh2`
            otherwise.
        lower_order_final (`bool`, default `True`):
            Whether to use lower-order solvers in the final steps. Only valid for < 15 inference steps. This can
            stabilize the sampling of DPMSolver for steps < 15, especially for steps <= 10.
        disable_corrector (`list`, default `[]`):
            Decides which step to disable the corrector to mitigate the misalignment between `epsilon_theta(x_t, c)`
            and `epsilon_theta(x_t^c, c)` which can influence convergence for a large guidance scale. Corrector is
            usually disabled during the first few steps.
        solver_p (`SchedulerMixin`, default `None`):
            Any other scheduler that if specified, the algorithm becomes `solver_p + UniC`.
        use_karras_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use Karras sigmas for step sizes in the noise schedule during the sampling process. If `True`,
            the sigmas are determined according to a sequence of noise levels {σi}.
        use_exponential_sigmas (`bool`, *optional*, defaults to `False`):
            Whether to use exponential sigmas for step sizes in the noise schedule during the sampling process.
        timestep_spacing (`str`, defaults to `"linspace"`):
            The way the timesteps should be scaled. Refer to Table 2 of the [Common Diffusion Noise Schedules and
            Sample Steps are Flawed](https://huggingface.co/papers/2305.08891) for more information.
        steps_offset (`int`, defaults to 0):
            An offset added to the inference steps, as required by some model families.
        final_sigmas_type (`str`, defaults to `"zero"`):
            The final `sigma` value for the noise schedule during the sampling process. If `"sigma_min"`, the final
            sigma is the same as the last sigma in the training schedule. If `zero`, the final sigma is set to 0.
    """

    _compatibles = [e.name for e in KarrasDiffusionSchedulers]
    order = 1

    @register_to_config
    def __init__(
            self,
            num_train_timesteps: int = 1000,
            solver_order: int = 2,
            prediction_type: str = "flow_prediction",
            shift: Optional[float] = 1.0,
            use_dynamic_shifting=False,
            thresholding: bool = False,
            dynamic_thresholding_ratio: float = 0.995,
            sample_max_value: float = 1.0,
            predict_x0: bool = True,
            solver_type: str = "bh2",
            lower_order_final: bool = True,
            disable_corrector: List[int] = [],
            solver_p: SchedulerMixin = None,
            timestep_spacing: str = "linspace",
            steps_offset: int = 0,
            final_sigmas_type: Optional[str] = "zero",  # "zero", "sigma_min"
    ):

        if solver_type not in ["bh1", "bh2"]:
            if solver_type in ["midpoint", "heun", "logrho"]:
                self.register_to_config(solver_type="bh2")
            else:
                raise NotImplementedError(
                    f"{solver_type} is not implemented for {self.__class__}")

        self.predict_x0 = predict_x0
        # setable values
        self.num_inference_steps = None
        alphas = np.linspace(1, 1 / num_train_timesteps,
                             num_train_timesteps)[::-1].copy()
        sigmas = 1.0 - alphas
        sigmas = torch.from_numpy(sigmas).to(dtype=torch.float32, device='cuda')

        if not use_dynamic_shifting:
            # when use_dynamic_shifting is True, we apply the timestep shifting on the fly based on the image resolution
            assert shift is not None, "shift must be not None when use_dynamic_shifting is False"
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        self.sigmas = sigmas
        self.timesteps = sigmas * num_train_timesteps

        self.model_outputs = [None] * solver_order
        self.timestep_list = [None] * solver_order
        self.lower_order_nums = 0
        self.disable_corrector = disable_corrector
        self.solver_p = solver_p
        self.last_sample = None

        self.sigma_min = self.sigmas[-1].item()
        self.sigma_max = self.sigmas[0].item()

    # Modified from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler.set_timesteps
    def set_timesteps(
        self,
        num_inference_steps: int,
        device: str | torch.device = None,
        sigmas: np.ndarray[float] | None = None,
        mu: float | None = None,
        shift: float | None = None,
    ):
        """
        Sets the discrete timesteps used for the diffusion chain (to be run before inference).
        Args:
            num_inference_steps (`int`):
                Total number of the spacing of the time steps.
            device (`str` or `torch.device`, *optional*):
                The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        """
        self.num_inference_steps = num_inference_steps

        if self.config.use_dynamic_shifting and mu is None:
            raise ValueError(
                " you have to pass a value for `mu` when `use_dynamic_shifting` is set to be `True`"
            )

        if sigmas is None:
            sigmas = np.linspace(
                self.sigma_max,
                self.sigma_min,
                num_inference_steps + 1,
            ).copy()[:-1]

        if self.config.use_dynamic_shifting:
            assert mu is not None
            sigmas = self.time_shift(mu, 1.0, sigmas)
        else:
            if shift is None:
                shift = self.config.shift
            assert isinstance(shift, float)
            assert sigmas is not None
            sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        if self.config.final_sigmas_type == "sigma_min":
            sigma_last = ((1 - self.alphas_cumprod[0]) / self.alphas_cumprod[0])**0.5
        elif self.config.final_sigmas_type == "zero":
            sigma_last = 0
        else:
            raise ValueError(
                f"`final_sigmas_type` must be one of 'zero', or 'sigma_min', but got {self.config.final_sigmas_type}"
            )

        timesteps = sigmas * self.config.num_train_timesteps
        sigmas = np.concatenate([sigmas, [sigma_last]]).astype(np.float32)

        self.sigmas = torch.from_numpy(sigmas).to(device=device)
        self.timesteps = torch.from_numpy(timesteps).to(device=device, dtype=torch.int64)

        self.model_outputs = [None] * self.config.solver_order
        self.lower_order_nums = 0
        self.last_sample = None
        if self.solver_p:
            self.solver_p.set_timesteps(self.num_inference_steps, device=device)

    # Copied from diffusers.schedulers.scheduling_ddpm.DDPMScheduler._threshold_sample
    def _threshold_sample(self, sample: torch.Tensor) -> torch.Tensor:
        """
        "Dynamic thresholding: At each sampling step we set s to a certain percentile absolute pixel value in xt0 (the
        prediction of x_0 at timestep t), and if s > 1, then we threshold xt0 to the range [-s, s] and then divide by
        s. Dynamic thresholding pushes saturated pixels (those near -1 and 1) inwards, thereby actively preventing
        pixels from saturation at each step. We find that dynamic thresholding results in significantly better
        photorealism as well as better image-text alignment, especially when using very large guidance weights."

        https://arxiv.org/abs/2205.11487
        """
        dtype = sample.dtype
        batch_size, channels, *remaining_dims = sample.shape

        if dtype not in (torch.float32, torch.float64):
            sample = sample.float(
            )  # upcast for quantile calculation, and clamp not implemented for cpu half

        # Flatten sample for doing quantile calculation along each image
        sample = sample.reshape(batch_size, channels * np.prod(remaining_dims))

        abs_sample = sample.abs()  # "a certain percentile absolute pixel value"

        s = torch.quantile(
            abs_sample, self.config.dynamic_thresholding_ratio, dim=1)
        s = torch.clamp(
            s, min=1, max=self.config.sample_max_value
        )  # When clamped to min=1, equivalent to standard clipping to [-1, 1]
        s = s.unsqueeze(
            1)  # (batch_size, 1) because clamp will broadcast along dim=0
        sample = torch.clamp(
            sample, -s, s
        ) / s  # "we threshold xt0 to the range [-s, s] and then divide by s"

        sample = sample.reshape(batch_size, channels, *remaining_dims)
        sample = sample.to(dtype)

        return sample

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.FlowMatchEulerDiscreteScheduler._sigma_to_t
    def _sigma_to_t(self, sigma):
        return sigma * self.config.num_train_timesteps

    @staticmethod
    def _sigma_to_alpha_sigma_t(sigma):
        return 1 - sigma, sigma

    # Copied from diffusers.schedulers.scheduling_flow_match_euler_discrete.set_timesteps
    def time_shift(self, mu: float, sigma: float, t: torch.Tensor):
        return math.exp(mu) / (math.exp(mu) + (1 / t - 1)**sigma)

    def convert_model_output(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
    ) -> torch.Tensor:
        r"""
        Convert the model output to the corresponding type the UniPC algorithm needs.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model.
            timestep (`int`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            step_index (`int`):
                The current timestep index.

        Returns:
            `torch.Tensor`:
                The converted model output.
        """
        if self.predict_x0:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[step_index]
                x0_pred = sample - sigma_t * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )

            if self.config.thresholding:
                x0_pred = self._threshold_sample(x0_pred)

            return x0_pred
        else:
            if self.config.prediction_type == "flow_prediction":
                sigma_t = self.sigmas[step_index]
                epsilon = sample - (1 - sigma_t) * model_output
            else:
                raise ValueError(
                    f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`,"
                    " `v_prediction` or `flow_prediction` for the UniPCMultistepScheduler."
                )

            if self.config.thresholding:
                sigma_t = self.sigmas[step_index]
                x0_pred = sample - sigma_t * model_output
                x0_pred = self._threshold_sample(x0_pred)
                epsilon = model_output + x0_pred

            return epsilon

    @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
    def multistep_uni_p_bh_update(
        self,
        model_output: torch.Tensor,
        sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        """
        One step for the UniP (B(h) version). Alternatively, `self.solver_p` is used if is specified.

        Args:
            model_output (`torch.Tensor`):
                The direct output from the learned diffusion model at the current timestep.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            order (`int`):
                The order of UniP at this timestep (corresponds to the *p* in UniPC-p).
            step_index (`int`):
                The current timestep index.

        Returns:
            `torch.Tensor`:
                The sample tensor at the previous timestep.
        """
        model_output_list = self.model_outputs

        s0 = self.timestep_list[-1]
        m0 = model_output_list[-1]
        x = sample

        if self.solver_p:
            x_t = self.solver_p.step(model_output, s0, x).prev_sample
            return x_t

        sigma_t, sigma_s0 = self.sigmas[step_index + 1], self.sigmas[step_index]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - i
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)  # (B, K)
            # for order 2, we use a simplified version
            if order == 2:
                rhos_p = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device)
            else:
                rhos_p = torch.linalg.solve_ex(R[:-1, :-1], b[:-1])[0].to(x.dtype)
        else:
            D1s = None
            rhos_p = None

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - alpha_t * B_h * pred_res
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                pred_res = torch.einsum("k,bkc...->bc...", rhos_p, D1s)
            else:
                pred_res = 0
            x_t = x_t_ - sigma_t * B_h * pred_res

        x_t = x_t.to(x.dtype)
        return x_t

    @torch.compile(mode="reduce-overhead", fullgraph=True, dynamic=False)
    def multistep_uni_c_bh_update(
        self,
        this_model_output: torch.Tensor,
        last_sample: torch.Tensor,
        this_sample: torch.Tensor,
        order: int,
        step_index: int,
    ) -> torch.Tensor:
        """
        One step for the UniC (B(h) version).

        Args:
            this_model_output (`torch.Tensor`):
                The model outputs at `x_t`.
            this_timestep (`int`):
                The current timestep `t`.
            last_sample (`torch.Tensor`):
                The generated sample before the last predictor `x_{t-1}`.
            this_sample (`torch.Tensor`):
                The generated sample after the last predictor `x_{t}`.
            order (`int`):
                The `p` of UniC-p at this step. The effective order of accuracy should be `order + 1`.
            step_index (`int`):
                The current timestep index.

        Returns:
            `torch.Tensor`:
                The corrected sample tensor at the current timestep.
        """
        model_output_list = self.model_outputs

        m0 = model_output_list[-1]
        x = last_sample
        x_t = this_sample
        model_t = this_model_output

        sigma_t, sigma_s0 = self.sigmas[step_index], self.sigmas[step_index - 1]
        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma_t)
        alpha_s0, sigma_s0 = self._sigma_to_alpha_sigma_t(sigma_s0)

        lambda_t = torch.log(alpha_t) - torch.log(sigma_t)
        lambda_s0 = torch.log(alpha_s0) - torch.log(sigma_s0)

        h = lambda_t - lambda_s0

        rks = []
        D1s = []
        for i in range(1, order):
            si = step_index - (i + 1)
            mi = model_output_list[-(i + 1)]
            alpha_si, sigma_si = self._sigma_to_alpha_sigma_t(self.sigmas[si])
            lambda_si = torch.log(alpha_si) - torch.log(sigma_si)
            rk = (lambda_si - lambda_s0) / h
            rks.append(rk)
            D1s.append((mi - m0) / rk)

        rks.append(torch.ones((), dtype=self.sigmas.dtype, device=self.sigmas.device))
        rks = torch.stack(rks, dim=0)

        R = []
        b = []

        hh = -h if self.predict_x0 else h
        h_phi_1 = torch.expm1(hh)  # h\phi_1(h) = e^h - 1
        h_phi_k = h_phi_1 / hh - 1

        factorial_i = 1

        if self.config.solver_type == "bh1":
            B_h = hh
        elif self.config.solver_type == "bh2":
            B_h = torch.expm1(hh)
        else:
            raise NotImplementedError()

        for i in range(1, order + 1):
            R.append(torch.pow(rks, i - 1))
            b.append(h_phi_k * factorial_i / B_h)
            factorial_i *= i + 1
            h_phi_k = h_phi_k / hh - 1 / factorial_i

        R = torch.stack(R, dim=0)
        b = torch.stack(b, dim=0)

        if len(D1s) > 0:
            D1s = torch.stack(D1s, dim=1)
        else:
            D1s = None

        # for order 1, we use a simplified version
        if order == 1:
            rhos_c = torch.full((1,), 0.5, dtype=x.dtype, device=self.sigmas.device)
        else:
            rhos_c = torch.linalg.solve_ex(R, b)[0].to(x.dtype)

        if self.predict_x0:
            x_t_ = sigma_t / sigma_s0 * x - alpha_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - alpha_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        else:
            x_t_ = alpha_t / alpha_s0 * x - sigma_t * h_phi_1 * m0
            if D1s is not None:
                corr_res = torch.einsum("k,bkc...->bc...", rhos_c[:-1], D1s)
            else:
                corr_res = 0
            D1_t = model_t - m0
            x_t = x_t_ - sigma_t * B_h * (corr_res + rhos_c[-1] * D1_t)
        x_t = x_t.to(x.dtype)
        return x_t

    def index_for_timestep(self, timestep, schedule_timesteps):
        indices = (schedule_timesteps == timestep).nonzero()

        # The sigma index that is taken for the **very** first `step`
        # is always the second index (or the last index if there is only 1)
        # This way we can ensure we don't accidentally skip a sigma in
        # case we start in the middle of the denoising schedule (e.g. for image-to-image)
        pos = 1 if len(indices) > 1 else 0

        return indices[pos].item()

    def step(
        self,
        model_output: torch.Tensor,
        timestep: torch.Tensor,
        sample: torch.Tensor,
        step_index: int,
        return_dict: bool = True,
    ) -> SchedulerOutput | tuple:
        """
        Predict the sample from the previous timestep by reversing the SDE. This function propagates the sample with
        the multistep UniPC.

        Args:
            model_output (`torch.Tensor`):
                The direct output from learned diffusion model.
            timestep (`torch.Tensor`):
                The current discrete timestep in the diffusion chain.
            sample (`torch.Tensor`):
                A current instance of a sample created by the diffusion process.
            step_index (`int`):
                The current timestep index.
            return_dict (`bool`):
                Whether or not to return a [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`.

        Returns:
            [`~schedulers.scheduling_utils.SchedulerOutput`] or `tuple`:
                If return_dict is `True`, [`~schedulers.scheduling_utils.SchedulerOutput`] is returned, otherwise a
                tuple is returned where the first element is the sample tensor.

        """
        if self.num_inference_steps is None:
            raise ValueError(
                "Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
            )

        use_corrector = (
            step_index > 0 and
            step_index - 1 not in self.disable_corrector and
            self.last_sample is not None
        )

        model_output_convert = self.convert_model_output(
            model_output=model_output,
            sample=sample,
            step_index=step_index,
        )
        if use_corrector:
            sample = self.multistep_uni_c_bh_update(
                this_model_output=model_output_convert,
                last_sample=self.last_sample,
                this_sample=sample,
                order=self.this_order,
                step_index=step_index,
            )
            # We must clone the outputs of a CUDA graph'd computation.
            sample = sample.clone()

        for i in range(self.config.solver_order - 1):
            self.model_outputs[i] = self.model_outputs[i + 1]
            self.timestep_list[i] = self.timestep_list[i + 1]

        self.model_outputs[-1] = model_output_convert
        self.timestep_list[-1] = timestep

        if self.config.lower_order_final:
            this_order = min(
                self.config.solver_order,
                len(self.timesteps) - step_index,
            )
        else:
            this_order = self.config.solver_order

        # Warmup for multistep.
        self.this_order = min(this_order, self.lower_order_nums + 1)
        assert self.this_order > 0

        self.last_sample = sample
        # Pass the original non-converted model output, in case solver-p is used.
        prev_sample = self.multistep_uni_p_bh_update(
            model_output=model_output,
            sample=sample,
            order=self.this_order,
            step_index=step_index,
        )
        # We must clone the outputs of a CUDA graph'd computation.
        prev_sample = prev_sample.clone()

        if self.lower_order_nums < self.config.solver_order:
            self.lower_order_nums += 1

        if not return_dict:
            return (prev_sample,)

        return SchedulerOutput(prev_sample=prev_sample)

    def scale_model_input(self, sample: torch.Tensor, *args,
                          **kwargs) -> torch.Tensor:
        """
        Ensures interchangeability with schedulers that need to scale the denoising model input depending on the
        current timestep.

        Args:
            sample (`torch.Tensor`):
                The input sample.

        Returns:
            `torch.Tensor`:
                A scaled input sample.
        """
        return sample

    # Copied from diffusers.schedulers.scheduling_dpmsolver_multistep.DPMSolverMultistepScheduler.add_noise
    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.IntTensor,
    ) -> torch.Tensor:
        # Make sure sigmas and timesteps have the same device and dtype as original_samples
        sigmas = self.sigmas.to(
            device=original_samples.device, dtype=original_samples.dtype)
        if original_samples.device.type == "mps" and torch.is_floating_point(timesteps):
            # mps does not support float64
            schedule_timesteps = self.timesteps.to(
                original_samples.device, dtype=torch.float32)
            timesteps = timesteps.to(
                original_samples.device, dtype=torch.float32)
        else:
            schedule_timesteps = self.timesteps.to(original_samples.device)
            timesteps = timesteps.to(original_samples.device)

        # begin_index is None when the scheduler is used for training or pipeline does not implement set_begin_index
        step_indices = [
            self.index_for_timestep(t, schedule_timesteps)
            for t in timesteps
        ]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < len(original_samples.shape):
            sigma = sigma.unsqueeze(-1)

        alpha_t, sigma_t = self._sigma_to_alpha_sigma_t(sigma)
        noisy_samples = alpha_t * original_samples + sigma_t * noise
        return noisy_samples

    def __len__(self):
        return self.config.num_train_timesteps
