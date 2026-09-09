from torch.autograd import Function
import torch

from modules.functional.backend import _backend

__all__ = ['trilinear_devoxelize']


def _gather_flat(features, indices):
    b, c, _ = features.shape
    return features.gather(2, indices.unsqueeze(1).expand(b, c, -1))


def _trilinear_devoxelize_torch(features, coords, resolution, is_training=True):
    b, c = features.shape[:2]
    r = int(resolution)
    r2 = r * r
    features = features.contiguous().view(b, c, -1)
    coords = coords.contiguous()

    x = coords[:, 0, :]
    y = coords[:, 1, :]
    z = coords[:, 2, :]
    x_lo = torch.floor(x).long()
    y_lo = torch.floor(y).long()
    z_lo = torch.floor(z).long()

    x_d1 = x - x_lo.to(dtype=x.dtype)
    y_d1 = y - y_lo.to(dtype=y.dtype)
    z_d1 = z - z_lo.to(dtype=z.dtype)
    x_d0 = 1.0 - x_d1
    y_d0 = 1.0 - y_d1
    z_d0 = 1.0 - z_d1

    x_hi = (x_d1 > 0).long() * r2
    y_hi = (y_d1 > 0).long() * r
    z_hi = (z_d1 > 0).long()

    idx000 = x_lo * r2 + y_lo * r + z_lo
    idx001 = idx000 + z_hi
    idx010 = idx000 + y_hi
    idx011 = idx010 + z_hi
    idx100 = idx000 + x_hi
    idx101 = idx100 + z_hi
    idx110 = idx100 + y_hi
    idx111 = idx110 + z_hi

    weights_and_indices = (
        (x_d0 * y_d0 * z_d0, idx000),
        (x_d0 * y_d0 * z_d1, idx001),
        (x_d0 * y_d1 * z_d0, idx010),
        (x_d0 * y_d1 * z_d1, idx011),
        (x_d1 * y_d0 * z_d0, idx100),
        (x_d1 * y_d0 * z_d1, idx101),
        (x_d1 * y_d1 * z_d0, idx110),
        (x_d1 * y_d1 * z_d1, idx111),
    )
    out = features.new_zeros(b, c, coords.shape[2])
    for weight, index in weights_and_indices:
        out = out + _gather_flat(features, index) * weight.unsqueeze(1)
    return out


class TrilinearDevoxelization(Function):
    @staticmethod
    def forward(ctx, features, coords, resolution, is_training=True):
        """
        :param ctx:
        :param coords: the coordinates of points, FloatTensor[B, 3, N]
        :param features: FloatTensor[B, C, R, R, R]
        :param resolution: int, the voxel resolution
        :param is_training: bool, training mode
        :return:
            FloatTensor[B, C, N]
        """
        B, C = features.shape[:2]
        features = features.contiguous().view(B, C, -1)
        coords = coords.contiguous()
        outs, inds, wgts = _backend.trilinear_devoxelize_forward(resolution, is_training, coords, features)
        if is_training:
            ctx.save_for_backward(inds, wgts)
            ctx.r = resolution
        return outs

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param ctx: 
        :param grad_output: gradient of outputs, FloatTensor[B, C, N]
        :return:
            gradient of inputs, FloatTensor[B, C, R, R, R]
        """
        inds, wgts = ctx.saved_tensors
        grad_inputs = _backend.trilinear_devoxelize_backward(grad_output.contiguous(), inds, wgts, ctx.r)
        return grad_inputs.view(grad_output.size(0), grad_output.size(1), ctx.r, ctx.r, ctx.r), None, None, None


trilinear_devoxelize = _trilinear_devoxelize_torch if _backend is None else TrilinearDevoxelization.apply
