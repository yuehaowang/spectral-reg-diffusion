import os
import warnings

from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
try:
    _backend = load(name='_pvcnn_backend',
                    extra_cflags=['-O3', '-std=c++17'],
                    # extra_cuda_cflags=['--compiler-bindir=/usr/bin/gcc'],
                    sources=[os.path.join(_src_path, 'src', f) for f in [
                        'ball_query/ball_query.cpp',
                        'ball_query/ball_query.cu',
                        'grouping/grouping.cpp',
                        'grouping/grouping.cu',
                        'interpolate/neighbor_interpolate.cpp',
                        'interpolate/neighbor_interpolate.cu',
                        'interpolate/trilinear_devox.cpp',
                        'interpolate/trilinear_devox.cu',
                        'sampling/sampling.cpp',
                        'sampling/sampling.cu',
                        'voxelization/vox.cpp',
                        'voxelization/vox.cu',
                        'bindings.cpp',
                    ]]
                    )
    _backend_load_error = None
except Exception as exc:
    _backend = None
    _backend_load_error = exc
    warnings.warn(
        f"Falling back to torch implementations for available PVCNN ops; "
        f"could not load _pvcnn_backend: {exc}",
        RuntimeWarning,
    )


def require_backend(op_name):
    if _backend is None:
        raise RuntimeError(
            f"{op_name} requires the _pvcnn_backend CUDA extension, but it "
            f"could not be loaded: {_backend_load_error}"
        )
    return _backend

__all__ = ['_backend']
