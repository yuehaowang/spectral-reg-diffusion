"""Compatibility fallback for the small subset of timm used by DiT-3D."""

from __future__ import annotations

import sys
import types

import torch.nn as nn


def ensure_timm_mlp() -> None:
    """Provide ``timm.models.vision_transformer.Mlp`` when timm is absent."""

    try:
        from timm.models.vision_transformer import Mlp as _Mlp  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    class Mlp(nn.Module):
        def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
        ):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.drop1 = nn.Dropout(drop)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.drop2 = nn.Dropout(drop)

        def forward(self, x):
            x = self.fc1(x)
            x = self.act(x)
            x = self.drop1(x)
            x = self.fc2(x)
            x = self.drop2(x)
            return x

    timm_module = sys.modules.setdefault("timm", types.ModuleType("timm"))
    models_module = sys.modules.setdefault("timm.models", types.ModuleType("timm.models"))
    vision_transformer_module = types.ModuleType("timm.models.vision_transformer")
    vision_transformer_module.Mlp = Mlp
    sys.modules["timm.models.vision_transformer"] = vision_transformer_module
    timm_module.models = models_module
    models_module.vision_transformer = vision_transformer_module
