from torchvision.models.googlenet import BasicConv2d, Inception

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: BasicConv2d, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: Inception, previous_shape: TensorShape) -> TensorShape:
    branch1_shape = compute_shape(module.branch1, previous_shape)
    branch2_shape = compute_shape(module.branch2, previous_shape)
    branch3_shape = compute_shape(module.branch3, previous_shape)
    branch4_shape = compute_shape(module.branch4, previous_shape)
    return TensorShape(
        previous_shape.height,
        previous_shape.width,
        branch1_shape.channels
        + branch2_shape.channels
        + branch3_shape.channels
        + branch4_shape.channels,
    )
