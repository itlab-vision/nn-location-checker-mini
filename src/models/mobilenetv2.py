import torch.nn as tnn
from torchvision.models.mobilenetv2 import InvertedResidual

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    channels = next(
        submodule.out_channels
        for submodule in reversed(list(module.conv))
        if isinstance(submodule, tnn.Conv2d)
    )
    return TensorShape(
        previous_shape.height // module.stride,
        previous_shape.height // module.stride,
        channels,
    )
