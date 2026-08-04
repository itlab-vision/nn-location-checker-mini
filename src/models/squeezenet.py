from torchvision.models.squeezenet import Fire

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: Fire, previous_shape: TensorShape) -> TensorShape:
    out_channels = module.expand1x1.out_channels + module.expand3x3.out_channels
    return TensorShape(previous_shape.height, previous_shape.width, out_channels)
