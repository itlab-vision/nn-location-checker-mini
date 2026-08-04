import torch.nn as tnn
from torchvision.models.mnasnet import _InvertedResidual

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: _InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    stride = next(
        m.stride[0]
        for m in module.layers.modules()
        if isinstance(m, tnn.Conv2d) and m.groups > 1
    )
    out_channels = next(
        m.out_channels
        for m in reversed(list(module.layers.modules()))
        if isinstance(m, tnn.Conv2d)
    )
    return TensorShape(
        previous_shape.height // stride, previous_shape.width // stride, out_channels
    )
