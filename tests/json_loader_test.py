import json
from pathlib import Path
from sys import path as sys_path

src_directory = Path(__file__).resolve().parents[1].joinpath("src")
sys_path.append(str(src_directory))

import pytest
import torch.nn as tnn

from classifier import Classifier
from json_loader import ModuleLoader
from tensor_shape import TensorShape


def write_json(tmp_path: Path, data: object, name: str = "modules.json") -> Path:
    path = tmp_path.joinpath(name)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


@pytest.fixture
def linear_json(tmp_path: Path):
    return write_json(tmp_path, [{"type": "linear", "out": 64}])


@pytest.fixture
def full_classifier_json(tmp_path: Path):
    return write_json(
        tmp_path,
        [
            {"type": "linear", "out": 128},
            {"type": "activation", "function": "relu"},
            {"type": "dropout", "percent": 0.5},
            {"type": "linear", "out": 15},
        ],
    )


def test_file_not_found_raises():
    with pytest.raises(ValueError, match="does not exist"):
        ModuleLoader("nonexistent.json")


def test_non_array_root_raises(tmp_path: Path):
    path = write_json(tmp_path, {"type": "linear", "out": 64})
    with pytest.raises(ValueError, match="array"):
        ModuleLoader(path)


def test_empty_array_loads(tmp_path: Path):
    path = write_json(tmp_path, [])
    loader = ModuleLoader(path)
    assert loader._modules == []


def test_unknown_module_type_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "unknown", "out": 64}])
    with pytest.raises(ValueError, match="Unknown module type"):
        ModuleLoader(path)


def test_load_returns_classifier(linear_json: Path):
    loader = ModuleLoader(linear_json)
    clf = loader.load(input_shape=128)
    assert isinstance(clf, Classifier)


def test_load_with_tensor_shape(linear_json: Path):
    loader = ModuleLoader(linear_json)
    clf = loader.load(input_shape=TensorShape(4, 4, 8))
    assert isinstance(clf, Classifier)
    assert clf.out_features == 64


def test_load_rewires_linear(linear_json: Path):
    loader = ModuleLoader(linear_json)
    clf = loader.load(input_shape=256)
    assert clf._layers[0].in_features == 256
    assert clf._layers[0].out_features == 64


def test_load_full_classifier(full_classifier_json: Path):
    loader = ModuleLoader(full_classifier_json)
    clf = loader.load(input_shape=512)
    assert clf.out_features == 15
    assert len(clf._layers) == 4


def test_linear_missing_out_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "linear"}])
    with pytest.raises(ValueError, match="'out'"):
        ModuleLoader(path)


def test_linear_default_bias(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "linear", "out": 32}])
    loader = ModuleLoader(path)
    assert loader._modules[0].bias is not None


def test_linear_no_bias(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "linear", "out": 32, "bias": False}])
    loader = ModuleLoader(path)
    assert loader._modules[0].bias is None


def test_activation_relu(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "activation", "function": "relu"}])
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.ReLU)


def test_activation_relu_inplace(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "activation", "function": "relu", "inplace": True}]
    )
    loader = ModuleLoader(path)
    assert loader._modules[0].inplace


def test_activation_unknown_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "activation", "function": "sigmoid"}])
    with pytest.raises(NotImplementedError, match="sigmoid"):
        ModuleLoader(path)


def test_activation_missing_function_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "activation"}])
    with pytest.raises(ValueError, match="'function'"):
        ModuleLoader(path)


def test_pool_max(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "pool", "function": "max", "kernel": 2, "stride": 2}]
    )
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.MaxPool2d)


def test_pool_avg(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "pool", "function": "avg", "kernel": 2, "stride": 2}]
    )
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.AvgPool2d)


def test_pool_unknown_raises(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "pool", "function": "min", "kernel": 2, "stride": 2}]
    )
    with pytest.raises(NotImplementedError, match="min"):
        ModuleLoader(path)


def test_pool_missing_kernel_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "pool", "function": "max", "stride": 2}])
    with pytest.raises(ValueError, match="'kernel'"):
        ModuleLoader(path)


def test_pool_missing_stride_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "pool", "function": "max", "kernel": 2}])
    with pytest.raises(ValueError, match="'stride'"):
        ModuleLoader(path)


def test_adaptive_pool_max(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "adaptive_pool", "function": "max", "out": [7, 7]}]
    )
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.AdaptiveMaxPool2d)


def test_adaptive_pool_avg(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "adaptive_pool", "function": "avg", "out": [7, 7]}]
    )
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.AdaptiveAvgPool2d)


def test_adaptive_pool_invalid_out_raises(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "adaptive_pool", "function": "avg", "out": 7}]
    )
    with pytest.raises(ValueError, match="tuple"):
        ModuleLoader(path)


def test_adaptive_pool_unknown_function_raises(tmp_path: Path):
    path = write_json(
        tmp_path, [{"type": "adaptive_pool", "function": "min", "out": [7, 7]}]
    )
    with pytest.raises(ValueError, match="min"):
        ModuleLoader(path)


def test_dropout_percent(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "dropout", "percent": 0.3}])
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.Dropout)
    assert loader._modules[0].p == pytest.approx(0.3)


def test_dropout_missing_percent_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "dropout"}])
    with pytest.raises(ValueError, match="'percent'"):
        ModuleLoader(path)


def test_dropout_inplace(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "dropout", "percent": 0.5, "inplace": True}])
    loader = ModuleLoader(path)
    assert loader._modules[0].inplace is True


def test_conv2d_built(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "convolution", "out": 64, "kernel": 3}])
    loader = ModuleLoader(path)
    assert isinstance(loader._modules[0], tnn.Conv2d)
    assert loader._modules[0].out_channels == 64


def test_conv2d_default_stride_and_padding(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "convolution", "out": 32, "kernel": 3}])
    loader = ModuleLoader(path)
    conv = loader._modules[0]
    assert conv.stride == (1, 1)
    assert conv.padding == (0, 0)


def test_conv2d_missing_kernel_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "convolution", "out": 64}])
    with pytest.raises(ValueError, match="'kernel'"):
        ModuleLoader(path)


def test_conv2d_missing_out_raises(tmp_path: Path):
    path = write_json(tmp_path, [{"type": "convolution", "kernel": 3}])
    with pytest.raises(ValueError, match="'out'"):
        ModuleLoader(path)
