from torchvision.models.resnet import BasicBlock, Bottleneck

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: BasicBlock, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for name, submodule in module.named_children():
        if name == "downsample":
            continue
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: Bottleneck, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for name, submodule in module.named_children():
        if name == "downsample":
            continue
        result_shape = compute_shape(submodule, result_shape)
    return result_shape
