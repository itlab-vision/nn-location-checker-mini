from torchvision.models.swin_transformer import (
    PatchMerging,
    PatchMergingV2,
    SwinTransformerBlock,
    SwinTransformerBlockV2,
)

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(
    _: SwinTransformerBlock | SwinTransformerBlockV2, previous_shape: TensorShape
) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(_: PatchMerging | PatchMergingV2, previous_shape: TensorShape) -> TensorShape:
    return TensorShape(
        previous_shape.height // 2,
        previous_shape.width // 2,
        previous_shape.channels * 2,
    )
