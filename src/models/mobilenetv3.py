import torch.nn as tnn
from torchvision.models.mobilenetv3 import InvertedResidual

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    stride = next(
        submodule.stride[0]
        for submodule in module.block.modules()
        if isinstance(submodule, tnn.Conv2d) and submodule.groups > 1
    )
    return TensorShape(
        previous_shape.height // stride,
        previous_shape.width // stride,
        module.out_channels,
    )
