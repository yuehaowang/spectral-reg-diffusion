import numpy as np
import torch

MODEL_CONFIGS = {
    "dit-xs": dict(patch_size=2, dim=64, n_layers=6, n_heads=4),
    "dit-s": dict(patch_size=2, dim=256, n_layers=10, n_heads=8),
    "dit-m": dict(patch_size=2, dim=256, n_layers=16, n_heads=32),
    "dit-l": dict(patch_size=2, dim=512, n_layers=16, n_heads=32),
    "dit-xl": dict(patch_size=2, dim=1024, n_layers=16, n_heads=16),
    "dit-xl4": dict(patch_size=4, dim=1024, n_layers=16, n_heads=16),
}


def create_dit_model(config, data_info):
    model_config = MODEL_CONFIGS[config.model_type]

    assert data_info.image_size >= 32

    if config.ssrepl:
        from .ssdit import DiT_Llama

        model = DiT_Llama(
            data_info.image_channels,
            data_info.image_size,
            num_classes=data_info.num_classes if config.cond else 0,
            encoder_depth=config.enc_depth,
            projector_dims=config.proj_dims,
            projector_bn=config.proj_bn,
            projector_adaln=config.proj_adaln,
            **model_config,
        )
    else:
        from .dit import DiT_Llama

        model = DiT_Llama(
            data_info.image_channels,
            data_info.image_size,
            num_classes=data_info.num_classes if config.cond else 0,
            **model_config,
        )

    return model
