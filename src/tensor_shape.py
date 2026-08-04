"""Tensor shape and computation for PyTorch modules.

This module provides `TensorShape` for representing tensor
dimensions and `compute_shape` for passing shapes through
`torch.nn.Module` layers without running a forward pass.
"""

from functools import singledispatch
from typing import NamedTuple, Never, overload

import torch.nn as tnn
from torchvision.ops import SqueezeExcitation
from torchvision.ops.misc import Permute

__all__ = ["TensorShape", "compute_conv", "compute_shape"]


class TensorShape(NamedTuple):
    height: int
    width: int
    channels: int

    def in_features(self) -> int:
        return self.height * self.channels * self.width


@overload
def _to_pair(obj: int | tuple[int, int]) -> tuple[int, int]: ...


@overload
def _to_pair(obj: tuple[int | None, int | None]) -> tuple[int | None, int | None]: ...


@overload
def _to_pair(obj: None) -> Never: ...


def _to_pair(obj: int | tuple[int, int] | tuple[int | None, int | None] | None):
    match obj:
        case int():
            return (obj, obj)
        case None:
            raise ValueError("Shape does not exist")
        case _:
            return obj


@overload
def compute_shape(module: tnn.Conv2d, previous_shape: TensorShape) -> TensorShape: ...


@overload
def compute_shape(
    _module: tnn.ReLU
    | tnn.Dropout
    | tnn.BatchNorm2d
    | tnn.ReLU6
    | tnn.Hardswish
    | tnn.SiLU
    | tnn.GELU,
    previous_shape: TensorShape,
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.MaxPool2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.AdaptiveAvgPool2d | tnn.AdaptiveMaxPool2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.Sequential | tnn.ModuleList, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(
    module: tnn.AvgPool2d, previous_shape: TensorShape
) -> TensorShape: ...


@overload
def compute_shape(module: tnn.Module, previous_shape: TensorShape) -> TensorShape: ...


def compute_conv(
    origin: int, padding: int, kernel_size: int, stride: int, dilation: int
) -> int:
    return (origin + 2 * padding - dilation * (kernel_size - 1) - 1) // stride + 1


@singledispatch
def compute_shape(module: tnn.Module, _previous_shape: TensorShape) -> Never:
    raise NotImplementedError(
        f"Cannot compute features map for the module: {type(module).__name__}"
    )


@compute_shape.register
def _(module: tnn.Conv2d, previous_shape: TensorShape) -> TensorShape:
    if isinstance(module.padding, str):
        raise NotImplementedError("Padding is string")
    new_height = compute_conv(
        previous_shape.height,
        module.padding[0],
        module.kernel_size[0],
        module.stride[0],
        module.dilation[0],
    )
    new_width = compute_conv(
        previous_shape.width,
        module.padding[1],
        module.kernel_size[1],
        module.stride[1],
        module.dilation[1],
    )
    return TensorShape(new_height, new_width, module.out_channels)


@compute_shape.register
def _(
    _module: tnn.ReLU
    | tnn.Dropout
    | tnn.BatchNorm2d
    | tnn.ReLU6
    | tnn.Hardswish
    | tnn.SiLU
    | tnn.GELU
    | tnn.Identity,
    previous_shape: TensorShape,
) -> TensorShape:
    return previous_shape


def compute_pool(
    origin: int, padding: int, kernel: int, stride: int, dilation: int, ceil_mode: bool
) -> int:
    dimension = origin + 2 * padding - dilation * (kernel - 1) - 1 + stride
    if ceil_mode:
        return (
            dimension + stride - 1
        ) // stride  # (x + y - 1) // y === ceil(x / y) for x > 0, y > 0
    return dimension // stride


@compute_shape.register
def _(module: tnn.MaxPool2d, previous_shape: TensorShape) -> TensorShape:
    padding_height, padding_width = _to_pair(module.padding)
    kernel_height, kernel_width = _to_pair(module.kernel_size)
    stride_height, stride_width = _to_pair(module.stride)
    dilation_height, dilation_width = _to_pair(module.dilation)

    new_height = compute_pool(
        previous_shape.height,
        padding_height,
        kernel_height,
        stride_height,
        dilation_height,
        module.ceil_mode,
    )
    new_width = compute_pool(
        previous_shape.width,
        padding_width,
        kernel_width,
        stride_width,
        dilation_width,
        module.ceil_mode,
    )
    return TensorShape(new_height, new_width, previous_shape.channels)


@compute_shape.register
def _(module: tnn.AvgPool2d, previous_shape: TensorShape) -> TensorShape:
    padding_height, padding_width = _to_pair(module.padding)
    kernel_height, kernel_width = _to_pair(module.kernel_size)
    stride_height, stride_width = _to_pair(module.stride)

    new_height = compute_pool(
        previous_shape.height,
        padding_height,
        kernel_height,
        stride_height,
        1,
        module.ceil_mode,
    )
    new_width = compute_pool(
        previous_shape.width,
        padding_width,
        kernel_width,
        stride_width,
        1,
        module.ceil_mode,
    )
    return TensorShape(new_height, new_width, previous_shape.channels)


@compute_shape.register
def _(
    module: tnn.AdaptiveAvgPool2d | tnn.AdaptiveMaxPool2d, previous_shape: TensorShape
) -> TensorShape:
    new_height, new_width = _to_pair(module.output_size)

    if new_height is None:
        new_height = previous_shape.height

    if new_width is None:
        new_width = previous_shape.width

    return TensorShape(
        new_height,
        new_width,
        previous_shape.channels,
    )


@compute_shape.register
def _(
    module: tnn.Sequential | tnn.ModuleList,
    previous_shape: TensorShape,
) -> TensorShape:
    result_shape = previous_shape
    for submodule in module:
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(_: tnn.LayerNorm, previous_shape: TensorShape) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(_: Permute, previous_shape: TensorShape) -> TensorShape:
    return previous_shape


@compute_shape.register
def _(_: SqueezeExcitation, previous_shape: TensorShape) -> TensorShape:
    return previous_shape
