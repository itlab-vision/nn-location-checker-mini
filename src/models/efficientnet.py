from torchvision.models.efficientnet import FusedMBConv, MBConv

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: MBConv | FusedMBConv, previous_shape: TensorShape) -> TensorShape:
    return compute_shape(module.block, previous_shape)
