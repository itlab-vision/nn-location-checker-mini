from torchvision.models.maxvit import (
    MaxVitBlock,
    MaxVitLayer,
    MBConv,
    PartitionAttentionLayer,
)

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: MBConv, previous_shape: TensorShape) -> TensorShape:
    return compute_shape(module.layers, previous_shape)


@compute_shape.register
def _(_: PartitionAttentionLayer, previous_shape: TensorShape) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(module: MaxVitLayer, previous_shape: TensorShape) -> TensorShape:
    return compute_shape(module.layers, previous_shape)


@compute_shape.register
def _(module: MaxVitBlock, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for layer in module.layers:
        result_shape = compute_shape(layer, result_shape)
    return result_shape
