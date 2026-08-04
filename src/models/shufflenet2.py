import torch.nn as tnn
from torchvision.models.shufflenetv2 import InvertedResidual

from tensor_shape import TensorShape, compute_shape


def _branch_channels(branch: tnn.Sequential) -> int:
    for module in reversed(branch):
        if isinstance(module, tnn.Conv2d):
            return module.out_channels
    raise ValueError("No Conv2d was detected in branch")


@compute_shape.register
def _(module: InvertedResidual, previous_shape: TensorShape) -> TensorShape:
    if module.stride == 1:
        return previous_shape
    channels = _branch_channels(module.branch1) + _branch_channels(module.branch2)
    return TensorShape(previous_shape.height // 2, previous_shape.width // 2, channels)
