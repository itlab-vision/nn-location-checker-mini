from torchvision.models.densenet import _DenseBlock, _DenseLayer, _Transition

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: _DenseLayer, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return TensorShape(
        result_shape.height,
        result_shape.width,
        previous_shape.channels + result_shape.channels,
    )


@compute_shape.register
def _(module: _DenseBlock, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for layer in module.children():
        result_shape = compute_shape(layer, result_shape)
    return result_shape


@compute_shape.register
def _(module: _Transition, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return result_shape
