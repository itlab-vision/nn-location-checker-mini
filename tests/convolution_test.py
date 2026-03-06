from pathlib import Path
from sys import path as sys_path

src_directory = (
    Path("./src/").absolute() if Path("./src/").exists() else Path("../src/").absolute()
)
sys_path.append(str(src_directory))

import torch.nn as tnn

from utils import TensorShape, compute_conv, compute_shape


class TestComputeConv:
    def test_conv(self):
        result = compute_conv(227, 2, 11, 4)
        assert result == 56


class TestComputShape:
    def test_conv(self):
        shape = TensorShape(32, 32, 3)
        padding = (2, 2)
        kernel_size = (5, 5)
        stride = (3, 3)
        conv = tnn.Conv2d(
            3, 64, padding=padding, kernel_size=kernel_size, stride=stride
        )
        result = compute_shape(conv, shape)

        checker_height = compute_conv(
            shape.height, padding[0], kernel_size[0], stride[0]
        )
        checker_width = compute_conv(shape.width, padding[1], kernel_size[1], stride[1])
        checker = TensorShape(checker_height, checker_width, 64)

        assert result == checker

    def test_relu(self):
        activation = tnn.ReLU()
        shape = TensorShape(32, 32, 3)

        result = compute_shape(activation, shape)

        assert result == shape

    def test_dropout(self):
        activation = tnn.Dropout(0.5)
        shape = TensorShape(32, 32, 3)

        result = compute_shape(activation, shape)

        assert result == shape

    def test_batch_norm(self):
        activation = tnn.BatchNorm2d(3024)
        shape = TensorShape(32, 32, 3)

        result = compute_shape(activation, shape)

        assert result == shape

    def test_max_pool(self):
        shape = TensorShape(32, 32, 3)
        padding = (2, 2)
        kernel_size = (5, 5)
        stride = (3, 3)
        pool = tnn.MaxPool2d(kernel_size[0], stride[0], padding[0])
        result = compute_shape(pool, shape)

        checker_height = compute_conv(
            shape.height, padding[0], kernel_size[0], stride[0]
        )
        checker_width = compute_conv(shape.width, padding[1], kernel_size[1], stride[1])
        checker = TensorShape(checker_height, checker_width, 3)

        assert result == checker

    def test_adaptive_max_pool(self):
        shape = TensorShape(32, 32, 3)
        out_size = (3, 3)
        pool = tnn.AdaptiveMaxPool2d(out_size)
        result = compute_shape(pool, shape)

        checker = TensorShape(out_size[0], out_size[1], 3)

        assert result == checker

    def test_adaptive_avg_pool(self):
        shape = TensorShape(32, 32, 3)
        out_size = (3, 3)
        pool = tnn.AdaptiveAvgPool2d(out_size)
        result = compute_shape(pool, shape)

        checker = TensorShape(out_size[0], out_size[1], 3)

        assert result == checker
