# Copyright 2025 Peng Sun
# Email: sunpeng@westlake.edu.cn
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from copy import deepcopy
import torch.nn.functional as F
from methods.transports import TRANSPORTS
from collections import OrderedDict


def mean_flat(x):
    """
    Take the mean over all non-batch dimensions.
    """
    return torch.mean(x, dim=list(range(1, len(x.size()))))


@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for model_name, param in model_params.items():
        if model_name in ema_params:
            ema_params[model_name].mul_(decay).add_(param.data, alpha=1 - decay)
        else:
            ema_name = (
                model_name.replace("module.", "")
                if model_name.startswith("module.")
                else f"module.{model_name}"
            )
            if ema_name in ema_params:
                ema_params[ema_name].mul_(decay).add_(param.data, alpha=1 - decay)
            else:
                raise KeyError(f"Parameter name {model_name} not found in EMA model!")


def get_attr_from_nested_module(model, attr_name):
    current = model
    while hasattr(current, "module"):
        current = current.module
    if hasattr(current, attr_name):
        return getattr(current, attr_name)
    else:
        return None


class UCGMTS(torch.nn.Module):
    """
    Unified Continuous Generative Model (UCGM).

    It unifies different training and generation paradigms, supporting:
    - Multi-step generation (akin to diffusion models or flow-matching models).
    - Few-step or single-step generation (akin to consistency models).

    Developed by: Peng Sun, email: sunpeng@westlake.edu.cn
    """

    def __init__(
        self,
        # --- Core Model & Transport Configuration ---
        transport_type: str = "Linear",
        # --- Training Strategy & Consistency Control ---
        consistc_ratio: float = 0.0,
        ema_decay_rate: float = 0.999,
        # --- Enhanced Target Score Mechanism ---
        enhanced_ratio: float = 0.0,
        lab_drop_ratio: float = 0.1,
        enhanced_range: list = [0.00, 0.75],
        # --- Loss Function Configuration ---
        scaled_cbl_eps: float = 0.0,
        wt_cosine_loss: bool = False,
        weight_funcion: str = None,
        # --- Time Discretization & Distribution ---
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
    ):
        """
        Initializes the UCGM model.

        Args:
            transport_type (str, optional): Specifies the type of transport mapping
                used in the model. Defaults to "Linear".

            consistc_ratio (float, optional): Consistency ratio, ranging from 0.0 to 1.0.
                This parameter interpolates between standard multi-step training and
                consistency-based few-step training.
                - 0.0: Standard multi-step training (e.g., for diffusion, flow-matching).
                - 1.0: Few-step/consistency model training.
                Defaults to 0.0.

            ema_decay_rate (float, optional): Decay rate for the Exponential Moving Average
                (EMA) of the model weights, used for the target network (stopgrad model).
                - ~0.999: Recommended for multi-step training.
                - 1.0: Recommended for few-step training, only when initializing
                         from a pre-trained multi-step model (implies using the online
                         model weights directly for the target).
                Defaults to 0.999.

            enhanced_ratio (float, optional): Ratio for incorporating enhanced target scores,
                ranging from 0.0 to 1.0. A value > 0 enables this mechanism.
                Higher values increase emphasis on these enhanced scores.
                Recommended range: [0.4, 0.7] for multi-step traing, [1.0, 2.0] for few-step training.
                Defaults to 0.0.

            lab_drop_ratio (float, optional): Dropout ratio for labels when using
                enhanced target scores. This promotes unconditional generation capabilities.
                Defaults to 0.1.

            enhanced_range (list[float], optional): Normalized time range [t_min, t_max]
                within which enhanced target scores are applied.
                Defaults to [0.00, 0.75].

            scaled_cbl_eps (float, optional): Hyperparameter for a scaled
                characteristic boundary-like loss. Modifies the loss landscape.
                Higher values make the loss behave more like an L2 loss.
                - 0.0: Recommended for multi-step models.
                - 5.0-9.0: Recommended for few-step models (can be model-dependent).
                Defaults to 0.0.

            wt_cosine_loss (bool, optional): If True, uses a cosine-similarity based
                weighting in the loss function.
                - True: Recommended for multi-step training.
                - False: Recommended for few-step training.
                Defaults to False.

            weight_function (str, optional): Specifies the type of weighting function
                to apply in the loss computation.
                - "Cosine": Recommended as a general-purpose choice.
                - None: No explicit weighting function beyond standard terms.
                Defaults to None.
                (Note: Original parameter name in code might be 'weight_funcion' due to a typo)

            time_dist_ctrl (list[float], optional): Parameters to control the
                distribution of time steps sampled during training. The interpretation
                depends on the specific sampling strategy implemented.
                - [1.0, 1.0, 1.0]: General multi-step training,
                    where [2.4, 2.4, 1.0] is better for SD-VAE-based training.
                - [0.8, 1.0, 1.0]: General few-step training.
                Defaults to [1.0, 1.0, 1.0].
        """
        super().__init__()
        self.tdr = lab_drop_ratio
        self.cor = consistc_ratio
        self.enr = enhanced_ratio
        self.huc = scaled_cbl_eps
        self.emd = ema_decay_rate
        self.eng = enhanced_range
        self.tdc = time_dist_ctrl
        self.wcl = wt_cosine_loss
        self.lwf = weight_funcion

        if self.enr >= 1.0 and self.cor == 0.0:
            self.enr = (self.enr - 1.0) / self.enr
            Warning("The enhance ratio larger than 1.0 is not supported")

        self.cmd = 0
        self.step = 0
        self.mod = None
        self.lsw = None

        transport = TRANSPORTS[transport_type]()
        self.alpha_in, self.gamma_in = transport.alpha_in, transport.gamma_in
        self.alpha_to, self.gamma_to = transport.alpha_to, transport.gamma_to

        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def sample_beta(self, alpha, beta, size):
        beta_dist = torch.distributions.Beta(alpha, beta)
        beta_samples = beta_dist.sample(size)
        return beta_samples

    def kumaraswamy_transform(self, t, a, b, c):
        return (1 - (1 - t**a) ** b) ** c

    def forward(self, model, x_t=None, t=None, **model_kwargs):
        dent = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        _t = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        F_t = model((x_t), _t, **model_kwargs)
        z_hat = (x_t * self.gamma_to(t) - F_t * self.gamma_in(t)) / dent
        x_hat = (F_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / dent
        return x_hat, z_hat, F_t, dent

    def loss_func(self, pd, pd_hat):
        loss = torch.sqrt(mean_flat((pd - pd_hat) ** 2) + self.huc**2) - self.huc
        loss += mean_flat(1 - F.cosine_similarity(pd, pd_hat, dim=1)) if self.wcl else 0
        return loss

    def enhance_target(self, target, idx, ndrop, pred_w_c, pred_wo_c):
        idx = idx[:-ndrop]
        target[:-ndrop][idx] = target[:-ndrop][idx] + self.enr * (
            pred_w_c[:-ndrop][idx] - pred_wo_c[:-ndrop][idx]
        )
        target[:-ndrop][~idx] = (target[:-ndrop][~idx] + pred_w_c[:-ndrop][~idx]) * 0.50
        target[-ndrop:] = (target[-ndrop:] + pred_wo_c[-ndrop:]) * 0.50
        return target

    def training_step(self, model, x=None, c=None):
        t = self.sample_beta(self.tdc[0], self.tdc[1], [x.size(0), 1, 1, 1]).to(x)
        t = torch.clamp(t * self.tdc[2], min=0, max=1)
        z = torch.randn_like(x)
        nullc = get_attr_from_nested_module(model, "num_classes")
        ndrop = round(self.tdr * len(c))
        c[-ndrop:] = nullc

        # Initialize target and model prediction
        x_t = z * self.alpha_in(t) + x * self.gamma_in(t)
        rng_state = torch.cuda.get_rng_state()
        x_wc_t, z_wc_t, F_t, den_t = self.forward(model, x_t, t, **dict(y=c))
        x_tar, z_tar, target = x, z, z * self.alpha_to(t) + x * self.gamma_to(t)

        if self.cor != 0.0 or self.enr != 0.0:
            with torch.no_grad():
                if self.emd > 0.0 and self.emd < 1.0:
                    self.mod = self.mod or deepcopy(model).requires_grad_(False).train()
                    update_ema(self.mod, model.module, decay=self.cmd)
                    self.cmd += (1 - self.cmd) * (self.emd - self.cmd) * 0.5
                elif self.emd == 0.0:
                    self.mod = model.module
                elif self.emd == 1.0:
                    self.mod = self.mod or deepcopy(model).requires_grad_(False).train()

                if self.enr != 0.0 and self.cor == 0.0:
                    # Get enhanced learning target w/o pre-trained model
                    e = torch.ones_like(c) * nullc
                    torch.cuda.set_rng_state(rng_state)
                    x_woc_t, z_woc_t, _, _ = self.forward(self.mod, x_t, t, **dict(y=e))
                    idx = (t.flatten() < self.eng[1]) & (t.flatten() > self.eng[0])
                    x_tar = self.enhance_target(x_tar, idx, ndrop, x_wc_t.data, x_woc_t)
                    z_tar = self.enhance_target(z_tar, idx, ndrop, z_wc_t.data, z_woc_t)
                    target = z_tar * self.alpha_to(t) + x_tar * self.gamma_to(t)

                if self.enr != 0.0 and self.cor != 0.0:
                    # Get enhanced learning target w/ pre-trained multi-step model
                    x_mwc_t, z_mwc_t, _, _ = self.forward(self.mod, x_t, t, **dict(y=c))
                    idx = (t.flatten() < self.eng[1]) & (t.flatten() > self.eng[0])
                    x_tar = self.enhance_target(x_tar, idx, ndrop, x_mwc_t, x_tar)
                    z_tar = self.enhance_target(z_tar, idx, ndrop, z_mwc_t, z_tar)
                    target = z_tar * self.alpha_to(t) + x_tar * self.gamma_to(t)

                if self.cor != 0.0:
                    # Calculate the value of f^x_t and f^x_{\lambda t}
                    def yfunc(r):
                        torch.cuda.set_rng_state(rng_state)
                        x_r = self.alpha_in(r) * z + self.gamma_in(r) * x
                        _, _, F_r, d_r = self.forward(model.module, x_r, r, **dict(y=c))
                        if self.enr != 0.0:
                            x_r = z_tar * self.alpha_in(r) + x_tar * self.gamma_in(r)
                        x_to_r = (F_r * self.alpha_in(r) - x_r * self.alpha_to(r)) / d_r
                        return x_to_r

                    # Calculate the derivative of f^x_t w.r.t. t
                    if self.cor == 1.0:
                        epsilo = 0.005
                        dv_dt = 1 / (2 * epsilo)
                        df_dv_dt = yfunc(t + epsilo) * dv_dt - yfunc(t - epsilo) * dv_dt
                    else:
                        epsilo = t - self.cor * t
                        dv_dt = 1 / epsilo
                        x_t = z_tar * self.alpha_in(t) + x_tar * self.gamma_in(t)
                        x_to_t = F_t.data * self.alpha_in(t) - x_t * self.alpha_to(t)
                        df_dv_dt = x_to_t / den_t * dv_dt - yfunc(t - epsilo) * dv_dt
                    # Calculate the learning target for F_{\theta}
                    df_dv_dt = torch.clamp(df_dv_dt, min=-1, max=1)
                    weight_w = 4 / torch.sin(t * 1.57)
                    target = F_t.data - (self.alpha_in(t) / den_t * weight_w) * df_dv_dt

        loss = self.loss_func(F_t, target)
        if self.lwf is None:
            return (loss).mean()
        elif self.lwf == "Cosine":
            return (loss * torch.cos(t * 1.57).flatten()).mean()

    @torch.no_grad()
    def uni_sample(
        self,
        inital_noise_z=None,
        sampling_model=None,
        sampling_steps: int = 35,
        stochast_ratio: float = 0.0,
        extrapol_ratio: float = 0.0,
        sampling_order: int = 1,
        time_dist_ctrl: list = [1.0, 1.0, 1.0],
        rfba_gap_steps: list = [0.001, 0.001],
        **model_kwargs,
    ):
        """
        Performs unified sampling to generate data samples from the learned distribution.

        Args:
            initial_noise_z (torch.Tensor, optional): Initial latent noise tensor (z_1).
                If None, sampled from model's prior (e.g., standard normal distribution).
                Defaults to None.

            sampling_model (torch.nn.Module, optional): Neural network model for predictions.
                If None, uses `self` (UCGMTS instance). Defaults to None.

            sampling_steps (int, optional): Number of discrete sampling steps.
                - 1 or 2: Few-step generation (consistency-like).
                - >20 and =<60: Multi-step generation (diffusion-like).
                Defaults to 35.

            stochastic_ratio (float, optional): Controls sampling stochasticity.
                - 0.0: Deterministic ODE-like path.
                - >0.0: Introduces stochasticity (SDE-like path).
                Recommended setting is the same value as consistc_ratio.
                Defaults to 0.0.

            extrapol_ratio (float, optional): Extrapolation ratio for accelerating sampling.
                Recommended range for multi-step models: [0.2, 0.6].
                - 0.0: Disables extrapolation.
                Defaults to 0.0.

            sampling_order (int, optional): Solver order for sampling.
                - 1: First-order prediction (Euler-Maruyama/DDIM-like).
                - 2: Second-order prediction/correction (Heun's method).
                Defaults to 1.

            time_dist_ctrl (list[float], optional): Kumaraswamy distribution parameters [a, b, c]
                for non-uniform timestep scheduling.
                - [1.17, 0.8, 1.1]: Recommended by UCGM.
                - [1.0, 1.0, 1.0]: Uniform/simplified distribution.
                Defaults to [1.0, 1.0, 1.0].

            rfba_gap_steps (list[float], optional): Controls the boundary offsets
                    [start_gap, end_gap] for timestep scheduling.
                Recommended configurations:
                - start_gap: Typically set to 0.0 or a small value like 0.001 (performance-dependent)
                - end_gap: Depends on model type:
                    * Pure multi-step models (consistc_ratio=0.0): 0.005 or smaller
                    * Pure few-step models (consistc_ratio=1.0): Between 0.2-0.8
                    * Hybrid models (0.0 < consistc_ratio < 1.0): Match end_gap to consistc_ratio value
                Defaults to [0.001, 0.001].

            **model_kwargs (Any): Additional arguments for model's forward pass.
                Used for conditioning etc.

        Notes:
            - Operates under `torch.no_grad()` context (gradients disabled).
            - Adapts behavior based on parameters to emulate multi-step (diffusion-like)
            or few-step (consistency-like) generation.
        """

        assert sampling_order in [1, 2]
        num_steps = (sampling_steps + 1) // 2 if sampling_order == 2 else sampling_steps

        # Time step discretization.
        num_steps = num_steps + 1 if (rfba_gap_steps[1] - 0.0) == 0.0 else num_steps
        t_steps = torch.linspace(
            rfba_gap_steps[0], 1.0 - rfba_gap_steps[1], num_steps, dtype=torch.float64
        ).to(inital_noise_z)
        t_steps = t_steps[:-1] if (rfba_gap_steps[1] - 0.0) == 0.0 else t_steps
        t_steps = self.kumaraswamy_transform(t_steps, *time_dist_ctrl)
        if self.integ_st:
            t_steps = 1 - t_steps
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

        # Prepare the buffer for the first order prediction.
        x_hats, z_hats, buffer_freq = [], [], 1

        # Main sampling loop.
        x_cur = inital_noise_z.to(torch.float64)
        samples = [inital_noise_z]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

            # First order prediction.
            x_hat, z_hat, _, _ = self.forward(
                sampling_model,
                x_cur.to(torch.float32),
                t_cur.to(torch.float32),
                **model_kwargs,
            )
            samples.append(x_hat)
            x_hat, z_hat = x_hat.to(torch.float64), z_hat.to(torch.float64)

            # Apply extrapolation for prediction (x is not nessary?).
            if buffer_freq > 0 and extrapol_ratio > 0:
                z_hats.append(z_hat)
                x_hats.append(x_hat)
                if i > buffer_freq:
                    z_hat = z_hat + extrapol_ratio * (z_hat - z_hats[-buffer_freq - 1])
                    x_hat = x_hat + extrapol_ratio * (x_hat - x_hats[-buffer_freq - 1])
                    z_hats.pop(0), x_hats.pop(0)

            if stochast_ratio == "Auto":
                stochast_ratio = (
                    torch.sqrt((t_next - t_cur).abs())
                    * torch.sqrt(2 * self.alpha_in(t_cur))
                    / self.alpha_in(t_next)
                )
                stochast_ratio = torch.clamp(stochast_ratio ** (1 / 0.5), min=0, max=1)

            x_next = self.gamma_in(t_next) * x_hat + self.alpha_in(t_next) * (
                z_hat * ((1 - stochast_ratio) ** 0.5)
                + torch.randn(x_cur.size()).to(x_cur) * (stochast_ratio**0.5)
            )

            # Apply second order correction.
            if sampling_order == 2 and i < num_steps - 1:
                x_pri, z_pri, _, _ = self.forward(
                    sampling_model,
                    x_next.to(torch.float32),
                    t_next.to(torch.float32),
                    **model_kwargs,
                )
                x_pri, z_pri = x_pri.to(torch.float64), z_pri.to(torch.float64)

                x_next = x_cur * self.gamma_in(t_next) / self.gamma_in(t_cur) + (
                    self.alpha_in(t_next)
                    - self.gamma_in(t_next)
                    * self.alpha_in(t_cur)
                    / self.gamma_in(t_cur)
                ) * (0.5 * z_hat + 0.5 * z_pri)

            x_cur = x_next

        return torch.stack(samples, dim=0).to(torch.float32)
