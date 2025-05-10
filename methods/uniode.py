import torch
from methods.transports import TRANSPORTS


class UniODE(torch.nn.Module):
    def __init__(self, transport_typ: str = "Linear"):
        super().__init__()

        transport = TRANSPORTS[transport_typ]()
        self.alpha_in, self.gamma_in = transport.alpha_in, transport.gamma_in
        self.alpha_to, self.gamma_to = transport.alpha_to, transport.gamma_to

        if self.gamma_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 0  # Start point if integral from 0 to 1
        elif self.alpha_in(torch.tensor(0)).abs().item() < 0.005:
            self.integ_st = 1  # Start point if integral from 1 to 0
        else:
            raise ValueError("Invalid Alpha and Gamma functions")

    def forward(self, model, x_t=None, t=None, **model_kwargs):
        den = self.alpha_in(t) * self.gamma_to(t) - self.gamma_in(t) * self.alpha_to(t)
        _t = torch.ones(x_t.size(0), device=x_t.device) * (t).flatten()
        f_t = model((x_t), _t, **model_kwargs)
        x_hat = (x_t * self.gamma_to(t) - f_t * self.gamma_in(t)) / den
        y_hat = (f_t * self.alpha_in(t) - x_t * self.alpha_to(t)) / den
        return y_hat, x_hat, f_t, den

    @torch.no_grad()
    def sample(
        self,
        x=None,
        model=None,
        num_steps=35,
        sampling_method="Euler",
        eta=0.001,
        **model_kwargs,
    ):
        assert sampling_method in ["Euler", "Heun"]

        if sampling_method == "Heun":
            num_steps = (num_steps + 1) // 2
        # Time step discretization.
        t_steps = torch.linspace(eta, 1.0 - eta, num_steps, dtype=torch.float64).to(x)
        t_steps = 1 - t_steps if self.integ_st else t_steps
        t_steps = torch.cat([torch.as_tensor(t_steps), torch.zeros_like(t_steps[:1])])

        # Main sampling loop.
        x_cur = x.to(torch.float64)
        samples = [x]
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):

            # Euler step.
            denoised, d_cur, _, _ = self.forward(
                model, x_cur.to(torch.float32), t_cur.to(torch.float32), **model_kwargs
            )
            samples.append(denoised)
            denoised, d_cur = denoised.to(torch.float64), d_cur.to(torch.float64)
            x_next = self.gamma_in(t_next) * denoised + self.alpha_in(t_next) * (d_cur)

            # Apply 2nd order correction.
            if sampling_method == "Heun" and i < num_steps - 1:
                denoised, d_pri, _, _ = self.forward(
                    model,
                    x_next.to(torch.float32),
                    t_next.to(torch.float32),
                    **model_kwargs,
                )
                denoised, d_pri = denoised.to(torch.float64), d_pri.to(torch.float64)
                x_next = x_cur * self.gamma_in(t_next) / self.gamma_in(t_cur) + (
                    self.alpha_in(t_next)
                    - self.gamma_in(t_next)
                    * self.alpha_in(t_cur)
                    / self.gamma_in(t_cur)
                ) * (0.5 * d_cur + 0.5 * d_pri)

            x_cur = x_next

        return torch.stack(samples, dim=0).to(torch.float32)
