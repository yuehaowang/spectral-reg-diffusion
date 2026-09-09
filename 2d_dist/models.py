"""Rectified-flow networks: a plain MLP and an SS-MLP with a probe-layer
projector ψ used by the spectral-regularization loss.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


def _broadcast_t(t, batch_size):
    """Ensure t is (batch_size, 1)."""
    if t.ndim == 1:
        t = t.reshape(-1, 1)
    if t.shape[0] == 1:
        t = t.expand(batch_size, 1)
    return t


class DiffusionMLP(nn.Module):
    """Baseline MLP that maps (t, x_t) -> velocity."""

    def __init__(self, in_dim=2, out_dim=2, hidden_dim=512, hidden_layers=8):
        super().__init__()
        layers = [nn.Linear(in_dim + 1, hidden_dim), nn.LeakyReLU()]
        for _ in range(hidden_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU()]
        layers += [nn.Linear(hidden_dim, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, t, x):
        t = _broadcast_t(t, x.shape[0])
        return self.mlp(torch.cat([x, t], dim=-1))


class NeuralEFProjector(nn.Module):
    """ψ(h, t): time-conditioned 2-layer MLP projecting probe-layer hidden
    states to a K-dim spectral feature.
    """

    def __init__(self, hidden_size, dims=(128, 64)):
        super().__init__()
        sizes = [hidden_size + 1] + list(dims)
        layers = []
        for i in range(len(sizes) - 2):
            layers += [
                nn.Linear(sizes[i], sizes[i + 1], bias=False),
                nn.BatchNorm1d(sizes[i + 1]),
                nn.ReLU(inplace=True),
            ]
        layers += [
            nn.Linear(sizes[-2], sizes[-1], bias=True),
            nn.BatchNorm1d(sizes[-1]),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, z, t):
        t = _broadcast_t(t, z.shape[0])
        return self.net(torch.cat([t, z], dim=-1))


class SS_DiffusionMLP(nn.Module):
    """MLP that also exposes a post-activation hidden state at ``probe_layer``
    after passing it through a learned projector ψ.
    """

    def __init__(self, in_dim=2, out_dim=2, hidden_dim=512, hidden_layers=8,
                 probe_layer=3, projector_dims=(128, 64)):
        super().__init__()
        # IMPORTANT: instantiate the projector BEFORE the MLP layers so the
        # global RNG state matches the original codebase, which built the
        # projector first inside create_model_and_optimizer.
        self.projector = NeuralEFProjector(hidden_dim, dims=projector_dims)
        self.probe_layer = probe_layer
        layers = [nn.Linear(in_dim + 1, hidden_dim)]
        for _ in range(hidden_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(nn.Linear(hidden_dim, out_dim))
        self.mlp = nn.ModuleList(layers)

    def forward(self, t, x, return_probed=False):
        t = _broadcast_t(t, x.shape[0])
        h = torch.cat([x, t], dim=-1)
        probed = None
        for i, layer in enumerate(self.mlp):
            h = layer(h)
            if i < len(self.mlp) - 1:
                h = F.leaky_relu(h)
            if return_probed and i == self.probe_layer:
                probed = self.projector(h, t=t)
        if return_probed:
            return h, probed
        return h
