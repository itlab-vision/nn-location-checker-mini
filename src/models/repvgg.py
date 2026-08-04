from timm.models.byobnet import RepVggBlock

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: RepVggBlock, previous_shape: TensorShape) -> TensorShape:
    if module.reparam_conv is not None:
        return compute_shape(module.reparam_conv, previous_shape)
    return compute_shape(module.conv_kxk.conv, previous_shape)
