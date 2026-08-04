from torchvision.models.convnext import CNBlock

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(_: CNBlock, previous_shape: TensorShape) -> TensorShape:
    return previous_shape
