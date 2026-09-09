"""Gaussian diffusion + a thin training wrapper around the DiT-3D backbone.

The diffusion class implements the standard variance-preserving forward
process and ancestral sampler from Ho et al. (DDPM). On top of that, when the
backbone exposes a projector ψ (the spectral-representation-regularized
variant), :meth:`GaussianDiffusion.p_losses` emits a spectral-regularization
term alongside the usual ε-MSE.

The ``Model`` wrapper just bundles the diffusion + backbone and exposes the
small handful of methods the train/test scripts need.
"""
import math

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn

from dit3d import DiT3D_models


# ---------------------------------------------------------------------------
#                             cross-rank AllReduce
# ---------------------------------------------------------------------------

class _AllReduceSum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(x, op=dist.ReduceOp.SUM)
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def _all_reduce_sum(x, enabled):
    return _AllReduceSum.apply(x) if enabled else x


# ---------------------------------------------------------------------------
#                             Diffusion process
# ---------------------------------------------------------------------------

def get_betas(schedule_type, b_start, b_end, time_num):
    if schedule_type == 'linear':
        return np.linspace(b_start, b_end, time_num)
    if schedule_type.startswith('warm'):
        frac = float(schedule_type[len('warm'):])
        betas = b_end * np.ones(time_num, dtype=np.float64)
        warm = int(time_num * frac)
        betas[:warm] = np.linspace(b_start, b_end, warm, dtype=np.float64)
        return betas
    raise ValueError(f"unknown schedule_type: {schedule_type!r}")


class GaussianDiffusion:
    """Discrete-time VP diffusion. Predicts ε, fixed-small variance.

    The SS-specific knobs are:
      * proj_loss_global_batch: broadcast t to all DDP ranks so the spectral
        cross-term is estimated from the global batch (needs ``dist`` init);
      * proj_single_t_batch: use a single t for the whole batch (a stronger
        estimator of the spectral matrix at a single timestep);
      * proj_triu_coeff: weight on the off-diagonal squared term;
      * proj_psi_scale: rescales ψ so the diagonal target is 2·proj_psi_scale.
    """

    def __init__(self, betas, *, loss_type='mse', model_mean_type='eps',
                 model_var_type='fixedsmall',
                 proj_loss_global_batch=True, proj_single_t_batch=False,
                 proj_triu_coeff=0.025, proj_psi_scale=1.0):
        assert isinstance(betas, np.ndarray)
        assert (betas > 0).all() and (betas <= 1).all()
        assert loss_type == 'mse' and model_mean_type == 'eps'
        assert model_var_type in ('fixedsmall', 'fixedlarge')
        self.loss_type = loss_type
        self.model_mean_type = model_mean_type
        self.model_var_type = model_var_type
        self.proj_loss_global_batch = proj_loss_global_batch
        self.proj_single_t_batch = proj_single_t_batch
        self.proj_triu_coeff = proj_triu_coeff
        self.proj_psi_scale = proj_psi_scale

        betas = betas.astype(np.float64)
        self.num_timesteps = int(betas.shape[0])
        alphas = 1.0 - betas
        ac = np.cumprod(alphas, axis=0)
        ac_prev = np.append(1.0, ac[:-1])
        self.betas = torch.from_numpy(betas).float()
        self.alphas = torch.from_numpy(alphas).float()
        self.alphas_cumprod = torch.from_numpy(ac).float()
        self.alphas_cumprod_prev = torch.from_numpy(ac_prev).float()
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1.0)
        post_var = self.betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_variance = post_var
        self.posterior_log_variance_clipped = torch.log(post_var.clamp(min=1e-20))
        self.posterior_mean_coef1 = (self.betas * torch.sqrt(self.alphas_cumprod_prev)
                                     / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coef2 = ((1.0 - self.alphas_cumprod_prev)
                                     * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod))

    @staticmethod
    def _extract(a, t, x_shape):
        out = torch.gather(a, 0, t)
        return out.reshape([x_shape[0]] + [1] * (len(x_shape) - 1))

    def _to(self, a, x):
        return a.to(x.device)

    def q_sample(self, x_start, t, noise):
        return (
            self._extract(self._to(self.sqrt_alphas_cumprod, x_start), t, x_start.shape) * x_start
            + self._extract(self._to(self.sqrt_one_minus_alphas_cumprod, x_start), t, x_start.shape) * noise
        )

    def _predict_xstart_from_eps(self, x_t, t, eps):
        return (
            self._extract(self._to(self.sqrt_recip_alphas_cumprod, x_t), t, x_t.shape) * x_t
            - self._extract(self._to(self.sqrt_recipm1_alphas_cumprod, x_t), t, x_t.shape) * eps
        )

    def _q_posterior_mean(self, x_start, x_t, t):
        return (
            self._extract(self._to(self.posterior_mean_coef1, x_t), t, x_t.shape) * x_start
            + self._extract(self._to(self.posterior_mean_coef2, x_t), t, x_t.shape) * x_t
        )

    def p_mean_variance(self, denoise_fn, x_t, t, y, clip_denoised):
        eps = denoise_fn(x_t, t, y)
        if self.model_var_type == 'fixedlarge':
            var = self.betas.to(x_t.device)
            log_var = torch.log(torch.cat([self.posterior_variance[1:2], self.betas[1:]])).to(x_t.device)
        else:
            var = self.posterior_variance.to(x_t.device)
            log_var = self.posterior_log_variance_clipped.to(x_t.device)
        var = self._extract(var, t, x_t.shape) * torch.ones_like(x_t)
        log_var = self._extract(log_var, t, x_t.shape) * torch.ones_like(x_t)
        x_recon = self._predict_xstart_from_eps(x_t, t, eps)
        if clip_denoised:
            x_recon = x_recon.clamp(-0.5, 0.5)
        mean = self._q_posterior_mean(x_recon, x_t, t)
        return mean, var, log_var

    # ----- sampling -----
    def p_sample(self, denoise_fn, x_t, t, y, noise_fn=torch.randn, clip_denoised=False):
        mean, _, log_var = self.p_mean_variance(denoise_fn, x_t, t, y, clip_denoised)
        noise = noise_fn(size=x_t.shape, dtype=x_t.dtype, device=x_t.device)
        nonzero = (1 - (t == 0).float()).reshape([x_t.shape[0]] + [1] * (x_t.ndim - 1))
        return mean + nonzero * torch.exp(0.5 * log_var) * noise

    @torch.no_grad()
    def p_sample_loop(self, denoise_fn, shape, device, y,
                      noise_fn=torch.randn, clip_denoised=True):
        x = noise_fn(size=shape, dtype=torch.float, device=device)
        for t in reversed(range(self.num_timesteps)):
            tb = torch.full((shape[0],), t, device=device, dtype=torch.int64)
            x = self.p_sample(denoise_fn, x, tb, y, noise_fn=noise_fn, clip_denoised=clip_denoised)
        return x

    @torch.no_grad()
    def p_sample_loop_trajectory(self, denoise_fn, shape, device, y, freq,
                                 noise_fn=torch.randn, clip_denoised=True):
        x = noise_fn(size=shape, dtype=torch.float, device=device)
        imgs = [x]
        for t in reversed(range(self.num_timesteps)):
            tb = torch.full((shape[0],), t, device=device, dtype=torch.int64)
            x = self.p_sample(denoise_fn, x, tb, y, noise_fn=noise_fn, clip_denoised=clip_denoised)
            if t % freq == 0 or t == self.num_timesteps - 1:
                imgs.append(x)
        return imgs

    # ----- training losses -----
    def _sample_t(self, B, device):
        if self.proj_single_t_batch:
            t = torch.randint(0, self.num_timesteps, size=(1,), device=device).repeat(B)
        else:
            t = torch.randint(0, self.num_timesteps, size=(B,), device=device)
        if self.proj_loss_global_batch and dist.is_available() and dist.is_initialized():
            dist.broadcast(t, src=0)
        return t

    def p_losses_mse(self, denoise_fn, x_start, y=None):
        """Plain ε-MSE loss (baseline DiT-w)."""
        B = x_start.shape[0]
        t = torch.randint(0, self.num_timesteps, size=(B,), device=x_start.device)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)
        eps = denoise_fn(x_t, t, y)
        return ((noise - eps) ** 2).mean(dim=list(range(1, x_start.ndim)))

    def p_losses_ss(self, denoise_fn, x_start, y=None):
        """ε-MSE + spectral representation regularization (ours).

        Each x_start is paired with itself (two independent noise draws). The
        projector outputs ψ1, ψ2 are normalized so the on-diagonal target is
        2·proj_psi_scale; off-diagonal triu-squared is regularized.
        """
        B = x_start.shape[0]
        device = x_start.device
        t = self._sample_t(B, device)

        # duplicate the batch (same x, same t, fresh noise) to get (ψ1, ψ2)
        x_start = torch.cat([x_start, x_start], dim=0)
        t = torch.cat([t, t], dim=0)
        if y is not None:
            y = torch.cat([y, y], dim=0)
        noise = torch.randn_like(x_start)
        x_t = self.q_sample(x_start, t, noise)

        eps, zs = denoise_fn(x_t, t, y, return_projector=True)
        sm_losses = ((noise - eps) ** 2).mean(dim=list(range(1, x_start.ndim)))

        # ψ ∈ (2B, L, K) -> per-token normalization across the (global) batch
        psi1, psi2 = zs.chunk(2)
        L, K = psi1.shape[1], psi1.shape[2]

        local_norm_sq = (psi1.pow(2) + psi2.pow(2)).sum(dim=0)  # (L, K)
        norm_sq = _all_reduce_sum(local_norm_sq, self.proj_loss_global_batch)
        norm = norm_sq.sqrt().clamp(min=1e-6)
        scale = math.sqrt(2.0 * self.proj_psi_scale)
        psi1 = psi1 / norm.view(1, L, -1) * scale
        psi2 = psi2 / norm.view(1, L, -1) * scale

        local_diag = (psi1 * psi2).sum(dim=0)
        diag = _all_reduce_sum(local_diag, self.proj_loss_global_batch)

        M1 = torch.triu(torch.einsum('blk,blm->lkm', psi1, psi2), diagonal=1)
        M2 = torch.triu(torch.einsum('blk,blm->lkm', psi2, psi1), diagonal=1)
        local_triu = M1.pow(2).sum(dim=(1, 2)) + M2.pow(2).sum(dim=(1, 2))
        triu = _all_reduce_sum(local_triu, self.proj_loss_global_batch)

        proj_loss = ((-diag.sum(-1) * 2) / K).mean() \
                    + self.proj_triu_coeff * (triu / K).mean()
        return sm_losses, proj_loss


# ---------------------------------------------------------------------------
#                             Model wrapper
# ---------------------------------------------------------------------------

class Model(nn.Module):
    """Bundle the DiT-3D backbone with the diffusion process and expose the
    small surface needed by ``train.py`` and ``eval.py``.

    Pass ``ss=True`` (plus ``encoder_depth``, ``projector_dims``, ...) to
    instantiate the spectral-representation-regularized variant.
    """

    def __init__(self, *, args, betas):
        super().__init__()
        proj_dims = tuple(args.proj_dims) if args.ss else ()
        self.ss = args.ss
        self.model = DiT3D_models[args.model_type](
            input_size=args.voxel_size,
            num_classes=args.num_classes,
            encoder_depth=args.encoder_depth,
            projector_dims=proj_dims,
            projector_bn=args.proj_bn,
            projector_adaln=args.proj_adaln,
        )
        self.diffusion = GaussianDiffusion(
            betas, loss_type=args.loss_type,
            model_mean_type=args.model_mean_type,
            model_var_type=args.model_var_type,
            proj_loss_global_batch=args.proj_global_batch,
            proj_single_t_batch=args.proj_single_t_batch,
            proj_triu_coeff=args.proj_triu_coeff,
        )

    # ε-prediction function used by the diffusion process.
    def _denoise(self, data, t, y, return_projector=False):
        return self.model(data, t, y, return_projector=return_projector)

    def get_loss_iter(self, data, y=None):
        if self.ss:
            return self.diffusion.p_losses_ss(self._denoise, data, y=y)
        return self.diffusion.p_losses_mse(self._denoise, data, y=y)

    def gen_samples(self, shape, device, y, noise_fn=torch.randn, clip_denoised=True):
        return self.diffusion.p_sample_loop(self._denoise, shape, device, y,
                                            noise_fn=noise_fn, clip_denoised=clip_denoised)

    def gen_sample_traj(self, shape, device, y, freq,
                       noise_fn=torch.randn, clip_denoised=True):
        return self.diffusion.p_sample_loop_trajectory(
            self._denoise, shape, device, y, freq=freq,
            noise_fn=noise_fn, clip_denoised=clip_denoised)

    def multi_gpu_wrapper(self, f):
        self.model = f(self.model)
