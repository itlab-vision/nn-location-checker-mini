from torchvision.models.inception import (
    BasicConv2d,
    InceptionA,
    InceptionB,
    InceptionC,
    InceptionD,
    InceptionE,
)

from tensor_shape import TensorShape, compute_pool, compute_shape


@compute_shape.register
def _(module: BasicConv2d, previous_shape: TensorShape) -> TensorShape:
    result_shape = previous_shape
    for submodule in module.children():
        result_shape = compute_shape(submodule, result_shape)
    return result_shape


@compute_shape.register
def _(module: InceptionA, previous_shape: TensorShape) -> TensorShape:
    b1 = compute_shape(module.branch1x1, previous_shape)
    b2 = compute_shape(module.branch5x5_1, previous_shape)
    b2 = compute_shape(module.branch5x5_2, b2)

    b3 = compute_shape(module.branch3x3dbl_1, previous_shape)
    b3 = compute_shape(module.branch3x3dbl_2, b3)
    b3 = compute_shape(module.branch3x3dbl_3, b3)
    b4 = compute_shape(module.branch_pool, previous_shape)

    return TensorShape(
        previous_shape.height,
        previous_shape.width,
        b1.channels + b2.channels + b3.channels + b4.channels,
    )


@compute_shape.register
def _(module: InceptionB, previous_shape: TensorShape) -> TensorShape:
    b1 = compute_shape(module.branch3x3, previous_shape)
    b2 = compute_shape(module.branch3x3dbl_1, previous_shape)
    b2 = compute_shape(module.branch3x3dbl_2, b2)
    b2 = compute_shape(module.branch3x3dbl_3, b2)
    b3_h = compute_pool(previous_shape.height, 0, 3, 2, 1, False)
    b3_w = compute_pool(previous_shape.width, 0, 3, 2, 1, False)

    return TensorShape(
        b3_h,
        b3_w,
        b1.channels + b2.channels + previous_shape.channels,
    )


@compute_shape.register
def _(module: InceptionC, previous_shape: TensorShape) -> TensorShape:
    b1 = compute_shape(module.branch1x1, previous_shape)
    b2 = compute_shape(module.branch7x7_1, previous_shape)
    b2 = compute_shape(module.branch7x7_2, b2)
    b2 = compute_shape(module.branch7x7_3, b2)

    b3 = compute_shape(module.branch7x7dbl_1, previous_shape)
    b3 = compute_shape(module.branch7x7dbl_2, b3)
    b3 = compute_shape(module.branch7x7dbl_3, b3)
    b3 = compute_shape(module.branch7x7dbl_4, b3)
    b3 = compute_shape(module.branch7x7dbl_5, b3)
    b4 = compute_shape(module.branch_pool, previous_shape)

    return TensorShape(
        previous_shape.height,
        previous_shape.width,
        b1.channels + b2.channels + b3.channels + b4.channels,
    )


@compute_shape.register
def _(module: InceptionD, previous_shape: TensorShape) -> TensorShape:
    b1 = compute_shape(module.branch3x3_1, previous_shape)
    b1 = compute_shape(module.branch3x3_2, b1)

    b2 = compute_shape(module.branch7x7x3_1, previous_shape)
    b2 = compute_shape(module.branch7x7x3_2, b2)
    b2 = compute_shape(module.branch7x7x3_3, b2)
    b2 = compute_shape(module.branch7x7x3_4, b2)
    pool_h = compute_pool(previous_shape.height, 0, 3, 2, 1, False)
    pool_w = compute_pool(previous_shape.width, 0, 3, 2, 1, False)

    return TensorShape(
        pool_h,
        pool_w,
        b1.channels + b2.channels + previous_shape.channels,
    )


@compute_shape.register
def _(module: InceptionE, previous_shape: TensorShape) -> TensorShape:
    b1 = compute_shape(module.branch1x1, previous_shape)
    b2_stem = compute_shape(module.branch3x3_1, previous_shape)
    b2a = compute_shape(module.branch3x3_2a, b2_stem)
    b2b = compute_shape(module.branch3x3_2b, b2_stem)

    b3_stem = compute_shape(module.branch3x3dbl_1, previous_shape)
    b3_stem = compute_shape(module.branch3x3dbl_2, b3_stem)
    b3a = compute_shape(module.branch3x3dbl_3a, b3_stem)
    b3b = compute_shape(module.branch3x3dbl_3b, b3_stem)
    b4 = compute_shape(module.branch_pool, previous_shape)

    return TensorShape(
        previous_shape.height,
        previous_shape.width,
        b1.channels
        + b2a.channels
        + b2b.channels
        + b3a.channels
        + b3b.channels
        + b4.channels,
    )
