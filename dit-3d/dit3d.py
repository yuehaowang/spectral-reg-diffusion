"""Plain DiT-3D backbone (baseline) plus an optional
spectral-representation-regularization projector ψ used by the ours variant.

When ``projector_dims`` is empty the projector is disabled and the model is
exactly the baseline DiT-3D from the original repo. Otherwise the forward
pass exposes the hidden state after block ``encoder_depth`` mapped through ψ
(a small MLP, optionally with BN and adaLN conditioning on (t, y)).

The single ``return_projector`` flag toggles the extra output, so a single
class serves both the baseline training run (DiT-w) and ours
(spectral-representation-regularized DiT-w).

Note on window attention: the upstream DiT-3D-WindAttn config exposed
``--window_size`` / ``--window_block_indexes`` flags but the original arg
parser used ``type=tuple`` on the indices, which silently turned the value
into a tuple of characters (``('0', ',', '3', ',', ...)``) that never
matched any integer block index. As a result every released checkpoint was
trained with effectively *global* attention. We accept the same flags in
``train.py`` and ``eval.py`` for checkpoint compatibility but ignore
them, so the released checkpoints reproduce exactly.
"""
import math

import numpy as np
import torch
import torch.nn as nn
from timm.models.vision_transformer import Mlp

from modules.voxelization import Voxelization
import modules.functional as MF


def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


# ---------------------------------------------------------------------------
#                           Embedding layers
# ---------------------------------------------------------------------------

class PatchEmbedVoxel(nn.Module):
    """Voxel -> patch embedding via a single 3D conv."""

    def __init__(self, voxel_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        self.num_patches = (voxel_size // patch_size) ** 3
        self.proj = nn.Conv3d(in_chans, embed_dim,
                              kernel_size=patch_size, stride=patch_size, bias=True)

    def forward(self, x):
        x = self.proj(x.float()).flatten(2).transpose(1, 2)
        return x


class TimestepEmbedder(nn.Module):
    """Sinusoidal timestep embedding -> 2-layer MLP."""

    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.frequency_embedding_size = frequency_embedding_size
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(0, half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
        return emb

    def forward(self, t):
        return self.mlp(self.timestep_embedding(t, self.frequency_embedding_size))


class LabelEmbedder(nn.Module):
    """Class-label embedding with classifier-free-guidance dropout."""

    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def _drop(self, labels, force_drop_ids=None):
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        return torch.where(drop_ids, self.num_classes, labels)

    def forward(self, labels, train, force_drop_ids=None):
        if (train and self.dropout_prob > 0) or force_drop_ids is not None:
            labels = self._drop(labels, force_drop_ids)
        return self.embedding_table(labels)


# ---------------------------------------------------------------------------
#                                Attention
# ---------------------------------------------------------------------------

class Attention(nn.Module):
    """3D multi-head self-attention (global over all patch tokens)."""

    def __init__(self, dim, num_heads=8, qkv_bias=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        # x: (B, X, Y, Z, C) — Y, Z dims are bookkeeping; attention runs over X*Y*Z
        B, X, Y, Z, _ = x.shape
        N = X * Y * Z
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, -1).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.reshape(3, B * self.num_heads, N, -1).unbind(0)
        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        x = (attn @ v).view(B, self.num_heads, X, Y, Z, -1
                            ).permute(0, 2, 3, 4, 1, 5).reshape(B, X, Y, Z, -1)
        return self.proj(x)


# ---------------------------------------------------------------------------
#                              DiT blocks
# ---------------------------------------------------------------------------

class DiTBlock(nn.Module):
    """adaLN-Zero DiT block (global attention)."""

    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, input_size=None):
        super().__init__()
        self.input_size = input_size
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads, qkv_bias=True)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.mlp = Mlp(in_features=hidden_size,
                       hidden_features=int(hidden_size * mlp_ratio),
                       act_layer=nn.GELU, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 6 * hidden_size, bias=True))

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = \
            self.adaLN_modulation(c).chunk(6, dim=1)
        B = x.shape[0]
        Xi, Yi, Zi = self.input_size
        shortcut = x

        h = modulate(self.norm1(x), shift_msa, scale_msa)
        h = self.attn(h.reshape(B, Xi, Yi, Zi, -1)).reshape(B, Xi * Yi * Zi, -1)
        x = shortcut + gate_msa.unsqueeze(1) * h
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class FinalLayer(nn.Module):
    """LayerNorm -> linear with adaLN-Zero conditioning."""

    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size ** 3 * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        return self.linear(modulate(self.norm_final(x), shift, scale))


# ---------------------------------------------------------------------------
#                       Spectral-regularization projector
# ---------------------------------------------------------------------------

class _BatchNorm1dSeq(nn.Module):
    """BN1d applied along the feature dim of a [B, L, C] token sequence."""

    def __init__(self, num_features):
        super().__init__()
        self.bn = nn.BatchNorm1d(num_features)

    def forward(self, x):
        return self.bn(x.transpose(-1, -2)).transpose(-1, -2)


class Projector(nn.Module):
    """ψ(z, c) — small MLP from the probed hidden state to a K-dim spectral
    feature, optionally with BN1d and adaLN(t, y) conditioning."""

    def __init__(self, hidden_size, dims, bn=True, adaln=False):
        super().__init__()
        self.use_adaln = adaln
        if not dims or dims[0] == 0:
            self.projector = nn.Identity()
        elif len(dims) == 1:
            self.projector = nn.Linear(hidden_size, dims[0], bias=True)
        else:
            sizes = [hidden_size] + list(dims)
            layers = []
            for i in range(len(sizes) - 2):
                layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
                if bn:
                    layers.append(_BatchNorm1dSeq(sizes[i + 1]))
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Linear(sizes[-2], sizes[-1], bias=True))
            if bn:
                layers.append(_BatchNorm1dSeq(sizes[-1]))
            self.projector = nn.Sequential(*layers)
        if adaln:
            self.adaLN_modulation = nn.Sequential(
                nn.SiLU(), nn.Linear(hidden_size, 2 * hidden_size, bias=True))
            self.norm = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)

    def forward(self, z, c=None):
        if self.use_adaln:
            shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
            z = modulate(self.norm(z), shift, scale)
        return self.projector(z)


# ---------------------------------------------------------------------------
#                                  DiT-3D
# ---------------------------------------------------------------------------

class DiT3D(nn.Module):
    """DiT-3D with window attention.

    Set ``projector_dims=[]`` (or pass ``return_projector=False``) for the
    baseline; non-empty ``projector_dims`` enables ψ and ``forward(...,
    return_projector=True)`` returns ``(eps, zs)``.
    """

    def __init__(self,
                 input_size=32, patch_size=4, in_channels=3,
                 hidden_size=1152, depth=28, num_heads=16, mlp_ratio=4.0,
                 class_dropout_prob=0.1, num_classes=1, learn_sigma=False,
                 encoder_depth=8,
                 projector_dims=(), projector_bn=False, projector_adaln=False):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.encoder_depth = encoder_depth

        self.voxelization = Voxelization(resolution=input_size, normalize=True, eps=0)
        self.x_embedder = PatchEmbedVoxel(input_size, patch_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbedder(hidden_size)
        self.y_embedder = LabelEmbedder(num_classes, hidden_size, class_dropout_prob)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.x_embedder.num_patches, hidden_size), requires_grad=False)

        n_per_axis = input_size // patch_size
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_size, num_heads, mlp_ratio=mlp_ratio,
                     input_size=(n_per_axis,) * 3)
            for _ in range(depth)
        ])
        self.projector = Projector(hidden_size, projector_dims,
                                   bn=projector_bn, adaln=projector_adaln)
        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)
        self._init_weights()

    # ---- init ------------------------------------------------------------
    def _init_weights(self):
        def _basic(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic)
        pos_embed = get_3d_sincos_pos_embed(
            self.pos_embed.shape[-1], self.input_size // self.patch_size)
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.normal_(self.y_embedder.embedding_table.weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

    # ---- forward ---------------------------------------------------------
    def _unpatchify(self, x0):
        c = self.out_channels
        p = self.patch_size
        nx = ny = nz = self.input_size // self.patch_size
        x0 = x0.reshape(x0.shape[0], nx, ny, nz, p, p, p, c)
        x0 = torch.einsum('nxyzpqrc->ncxpyqzr', x0)
        return x0.reshape(x0.shape[0], c, nx * p, ny * p, nz * p)

    def forward(self, x, t, y, return_projector=False):
        # x: (B, 3, N) point cloud, returns (B, 3, N) predicted eps.
        x, voxel_coords = self.voxelization(x, x)
        x = self.x_embedder(x) + self.pos_embed
        c = self.t_embedder(t) + self.y_embedder(y, self.training)

        zs = None
        for i, block in enumerate(self.blocks):
            x = block(x, c)
            if return_projector and (i + 1) == self.encoder_depth:
                zs = self.projector(x, c)

        x = self.final_layer(x, c)
        x = self._unpatchify(x)
        x = MF.trilinear_devoxelize(x, voxel_coords, self.input_size, self.training)
        return (x, zs) if return_projector else x


# ---------------------------------------------------------------------------
#               Sin/cos positional embedding (MAE-style, 3D)
# ---------------------------------------------------------------------------

def _get_1d_sincos_pos_embed(embed_dim, pos):
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64) / (embed_dim / 2.0)
    omega = 1.0 / 10000 ** omega
    pos = pos.reshape(-1)
    out = np.einsum("m,d->md", pos, omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


def get_3d_sincos_pos_embed(embed_dim, grid_size):
    assert embed_dim % 3 == 0
    grid = np.meshgrid(*([np.arange(grid_size, dtype=np.float32)] * 3), indexing="ij")
    grid = np.stack(grid, axis=0).reshape(3, 1, grid_size, grid_size, grid_size)
    parts = [_get_1d_sincos_pos_embed(embed_dim // 3, grid[i]) for i in range(3)]
    return np.concatenate(parts, axis=1)


# ---------------------------------------------------------------------------
#                              Model configs
# ---------------------------------------------------------------------------

def _dit(depth, hidden, patch, heads):
    def builder(**kwargs):
        return DiT3D(depth=depth, hidden_size=hidden, patch_size=patch,
                     num_heads=heads, **kwargs)
    return builder


DiT3D_models = {
    "DiT-XL/2": _dit(28, 1152, 2, 16),
    "DiT-XL/4": _dit(28, 1152, 4, 16),
    "DiT-XL/8": _dit(28, 1152, 8, 16),
    "DiT-L/2":  _dit(24, 1152, 2, 16),
    "DiT-L/4":  _dit(24, 1152, 4, 16),
    "DiT-L/8":  _dit(24, 1152, 8, 16),
    "DiT-B/2":  _dit(12, 768,  2, 12),
    "DiT-B/4":  _dit(12, 768,  4, 12),
    "DiT-B/8":  _dit(12, 768,  8, 12),
    "DiT-S/2":  _dit(12, 384,  2, 6),
    "DiT-S/4":  _dit(12, 384,  4, 6),
    "DiT-S/8":  _dit(12, 384,  8, 6),
}
