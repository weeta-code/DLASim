"""
Cross-channel-consistent GaussianDiffusion wrapper.

Adds an auxiliary loss term that penalizes per-pixel inconsistency
between the binary-presence channel (ch0) and the auxiliary channels
(ch1, ch2, ch3). Specifically, where ch0 is small (no particle), the
auxiliary channels should also be small.

Loss form:
    L_total = L_diffusion + lambda * L_consistency
    L_consistency = mean[ (1 - ch0_pred)^2 * sum_k (ch_k_pred^2) ]   k = 1, 2, 3

This term is differentiable w.r.t. the model output, scale-free in the
data range [0, 1], and zero when ch0_pred ≈ 1 (particle present, free
to assign any auxiliary value) OR when ch_k_pred ≈ 0 (no particle and
no auxiliary signal — consistent).

Targets the "forest issue": v4mc's tendency to scatter small disjoint
activations is partly because nothing prevents the model from emitting
low-amplitude ch0 spikes that aren't backed by consistent ch1/2/3
values. A consistency penalty makes these isolated spikes expensive.

Built as a thin wrapper around lucidrains' GaussianDiffusion so we can
keep all the existing v-prediction / cosine / min-SNR machinery.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossChannelGaussianDiffusion(nn.Module):
    """Wraps a GaussianDiffusion instance and adds cross-channel consistency.

    Forwarding `(batch)` returns the combined loss. All other GaussianDiffusion
    methods (`sample`, `state_dict`, etc.) are forwarded transparently.
    """

    def __init__(self, base_diffusion, weight: float = 0.1,
                 mode: str = "presence"):
        super().__init__()
        self.base = base_diffusion
        self.weight = weight
        self.mode = mode  # 'presence' or 'rgb'

    # ----- Cross-channel consistency on predicted x_0 -----
    @staticmethod
    def consistency_loss_presence(x_0_pred: torch.Tensor) -> torch.Tensor:
        """
        For "presence" encoding (channel 0 = binary mask, others auxiliary):
            penalty = (1 - ch0)^2 * sum_k(ch_k^2) for k >= 1
        Pushes auxiliary channels to zero where ch0 is low.
        """
        if x_0_pred.shape[1] < 2:
            return torch.tensor(0.0, device=x_0_pred.device,
                                dtype=x_0_pred.dtype)
        ch0 = x_0_pred[:, 0:1].clamp(0.0, 1.0)
        ch_aux = x_0_pred[:, 1:].clamp(0.0, 1.0)
        suppression = (1.0 - ch0) ** 2
        penalty = (ch_aux ** 2) * suppression
        return penalty.mean()

    @staticmethod
    def consistency_loss_rgb(x_0_pred: torch.Tensor) -> torch.Tensor:
        """
        For RGB seed-blue encoding (R=G for non-seed, B=255 only at seed):

          term_rg   = (R - G)^2 * (1 - B)^2
              penalize R != G except where B is high (the seed),
              since real particles satisfy R == G by construction.

          term_blue = B * R * G
              penalize coexistence of strong B with strong R or G
              (the seed disc has R=G=0 and B=1; non-seed pixels have B=0).

        We deliberately avoid a sparsity term: in this encoding,
        particles near the seed are LEGITIMATELY dim (R=G ~ 0 because
        distance ~ 0), so suppressing weak activations would punish
        valid signal.
        """
        x = x_0_pred.clamp(0.0, 1.0)
        if x.shape[1] != 3:
            # Fall back to presence-style loss for non-RGB inputs
            return CrossChannelGaussianDiffusion.consistency_loss_presence(
                x_0_pred)

        R = x[:, 0:1]
        G = x[:, 1:2]
        B = x[:, 2:3]

        # 1) R must equal G except where B is large (seed)
        term_rg = ((R - G) ** 2 * (1.0 - B) ** 2).mean()

        # 2) Strong B and strong R/G shouldn't coexist
        term_blue = (B * R * G).mean()

        return term_rg + 0.5 * term_blue

    @classmethod
    def consistency_loss(cls, x_0_pred: torch.Tensor,
                         mode: str = "presence") -> torch.Tensor:
        """Dispatch to the appropriate per-mode consistency loss."""
        if mode == "rgb":
            return cls.consistency_loss_rgb(x_0_pred)
        return cls.consistency_loss_presence(x_0_pred)

    # ----- Recover predicted x_0 given the model output -----
    def _recover_x0(self, x_t, t, model_out, objective):
        """Returns predicted x_0 in the data range."""
        # GaussianDiffusion stores buffers like sqrt_alphas_cumprod, etc.
        gd = self.base
        sqrt_alphas_cumprod_t = gd.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = (
            gd.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        )
        if objective == "pred_v":
            # v = sqrt(alpha_bar)*eps - sqrt(1 - alpha_bar)*x_0
            # x_0 = sqrt(alpha_bar)*x_t - sqrt(1 - alpha_bar)*v
            return (sqrt_alphas_cumprod_t * x_t
                    - sqrt_one_minus_alphas_cumprod_t * model_out)
        elif objective == "pred_noise":
            # eps = (x_t - sqrt(alpha_bar)*x_0) / sqrt(1 - alpha_bar)
            # x_0 = (x_t - sqrt(1 - alpha_bar)*eps) / sqrt(alpha_bar)
            return ((x_t - sqrt_one_minus_alphas_cumprod_t * model_out)
                    / sqrt_alphas_cumprod_t)
        elif objective == "pred_x0":
            return model_out
        else:
            raise ValueError(f"Unknown objective: {objective}")

    def forward(self, batch):
        """Compute total loss = diffusion loss + λ * consistency loss."""
        gd = self.base
        b, c, h, w = batch.shape
        device = batch.device
        t = torch.randint(0, gd.num_timesteps, (b,), device=device).long()
        noise = torch.randn_like(batch)
        x_t = gd.q_sample(x_start=batch, t=t, noise=noise)

        model_out = gd.model(x_t, t)

        # ---- Standard diffusion loss (replicates GaussianDiffusion.p_losses) ----
        if gd.objective == "pred_v":
            target = gd.predict_v(batch, t, noise)
        elif gd.objective == "pred_noise":
            target = noise
        elif gd.objective == "pred_x0":
            target = batch
        else:
            raise ValueError(f"Unknown objective: {gd.objective}")

        loss_per_elem = F.mse_loss(model_out, target, reduction="none")
        loss_per_elem = loss_per_elem.mean(dim=[1, 2, 3])

        # min-SNR loss weighting if enabled
        if getattr(gd, "min_snr_loss_weight", False) is not False and gd.min_snr_loss_weight:
            # try the buffer if registered
            try:
                snr = gd.alphas_cumprod[t] / (1.0 - gd.alphas_cumprod[t])
                # min-SNR-gamma weighting (typical gamma=5)
                min_snr_gamma = getattr(gd, "min_snr_gamma", 5.0)
                clipped = snr.clamp(max=min_snr_gamma)
                if gd.objective == "pred_noise":
                    weight = clipped / snr
                elif gd.objective == "pred_v":
                    weight = clipped / (snr + 1.0)
                elif gd.objective == "pred_x0":
                    weight = clipped
                else:
                    weight = torch.ones_like(snr)
                loss_diffusion = (loss_per_elem * weight).mean()
            except (AttributeError, KeyError):
                loss_diffusion = loss_per_elem.mean()
        else:
            loss_diffusion = loss_per_elem.mean()

        # ---- Cross-channel consistency on predicted x_0 ----
        with torch.no_grad():
            x_0_pred = self._recover_x0(x_t, t, model_out.detach(),
                                        gd.objective)
        # Re-enable gradients via the model_out path:
        # We need to backprop through the consistency loss, so recompute
        # x_0_pred WITH grad through model_out.
        x_0_pred = self._recover_x0(x_t, t, model_out, gd.objective)

        cc_loss = self.consistency_loss(x_0_pred, mode=self.mode)

        return loss_diffusion + self.weight * cc_loss

    # ----- Pass-through methods so EMA / sampling work as before -----
    def __getattr__(self, name):
        # When pytorch hasn't bound the attribute yet, fall back to the wrapped
        # diffusion. This allows model.sample(...), model.parameters(), etc.
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self._modules["base"], name)
