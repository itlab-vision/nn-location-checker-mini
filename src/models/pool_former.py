from timm.models.metaformer import Downsampling, MetaFormerBlock, MetaFormerStage, Stem

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(module: Stem, previous_shape: TensorShape) -> TensorShape:
    return compute_shape(module.conv, previous_shape)


@compute_shape.register
def _(module: Downsampling, previous_shape: TensorShape) -> TensorShape:
    return compute_shape(module.conv, previous_shape)


@compute_shape.register
def _(_: MetaFormerBlock, previous_shape: TensorShape) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(module: MetaFormerStage, previous_shape: TensorShape) -> TensorShape:
    result_shape = compute_shape(module.downsample, previous_shape)
    return compute_shape(module.blocks, result_shape)
