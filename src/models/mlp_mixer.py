from timm.layers import PatchEmbed
from timm.models.mlp_mixer import MixerBlock

from tensor_shape import TensorShape, compute_shape


@compute_shape.register
def _(_: MixerBlock, previous_shape: TensorShape) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(module: PatchEmbed, previous_shape: TensorShape) -> TensorShape:
    patch_shape = compute_shape(module.proj, previous_shape)
    seq_length = patch_shape.height * patch_shape.width
    return TensorShape(1, seq_length, patch_shape.channels)
