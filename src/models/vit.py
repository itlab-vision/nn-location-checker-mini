from typing import override

import torch
import torch.nn as tnn
from torchvision.models.vision_transformer import Encoder

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(_: Encoder, previous_shape: TensorShape) -> TensorShape:
    return previous_shape


class ViTPatch(tnn.Module):
    def __init__(self, conv_proj: tnn.Conv2d, class_token: tnn.Parameter) -> None:
        super().__init__()
        self.conv_proj = conv_proj
        self.class_token = class_token

    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = x.shape[0]
        x = self.conv_proj(x)
        x = x.flatten(2).permute(0, 2, 1)
        batch_class_token = self.class_token.expand(n, -1, -1)
        return torch.cat([batch_class_token, x], dim=1)


@compute_shape.register
def _(module: ViTPatch, previous_shape: TensorShape) -> TensorShape:
    patch_shape = compute_shape(module.conv_proj, previous_shape)
    seq_length = patch_shape.height * patch_shape.width + 1
    return TensorShape(1, seq_length, patch_shape.channels)
