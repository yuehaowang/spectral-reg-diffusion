from torch.autograd import Function
import torch

from modules.functional.backend import _backend

__all__ = ['avg_voxelize']


def _avg_voxelize_torch(features, coords, resolution):
    b, c, n = features.shape
    r = int(resolution)
    r2 = r * r
    r3 = r2 * r
    coords = coords.long()
    indices = coords[:, 0, :] * r2 + coords[:, 1, :] * r + coords[:, 2, :]

    out = features.new_zeros(b, c, r3)
    out.scatter_add_(2, indices.unsqueeze(1).expand(-1, c, -1), features)

    counts = features.new_zeros(b, 1, r3)
    counts.scatter_add_(2, indices.unsqueeze(1), torch.ones(b, 1, n, device=features.device, dtype=features.dtype))
    out = out / counts.clamp_min(1.0)
    return out.view(b, c, r, r, r)


class AvgVoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution):
        """
        :param ctx:
        :param features: Features of the point cloud, FloatTensor[B, C, N]
        :param coords: Voxelized Coordinates of each point, IntTensor[B, 3, N]
        :param resolution: Voxel resolution
        :return:
            Voxelized Features, FloatTensor[B, C, R, R, R]
        """
        features = features.contiguous()
        coords = coords.int().contiguous()
        b, c, _ = features.shape
        out, indices, counts = _backend.avg_voxelize_forward(features, coords, resolution)
        ctx.save_for_backward(indices, counts)
        return out.view(b, c, resolution, resolution, resolution)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx:
        :param grad_output: gradient of output, FloatTensor[B, C, R, R, R]
        :return:
            gradient of inputs, FloatTensor[B, C, N]
        """
        b, c = grad_output.shape[:2]
        indices, counts = ctx.saved_tensors
        grad_features = _backend.avg_voxelize_backward(grad_output.contiguous().view(b, c, -1), indices, counts)
        return grad_features, None, None


avg_voxelize = _avg_voxelize_torch if _backend is None else AvgVoxelization.apply
