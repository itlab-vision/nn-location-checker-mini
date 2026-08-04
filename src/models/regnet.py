from torchvision.models.regnet import ResBottleneckBlock

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: ResBottleneckBlock, previous_shape: TensorShape) -> TensorShape:
    return compute_shape(module.f, previous_shape)
