"""Model registry: enums and factory for donor model loading.

Enums are pure named constants. Loading logic lives in `load_model_internals`,
which returns a `ModelInternals` dataclass — modules and optional transform —
in a single call.
"""

from dataclasses import dataclass
from enum import Enum
from typing import NamedTuple

import open_clip
import timm
import torch.nn as tnn
import torchvision.models as tvm
import torchvision.transforms.v2 as tt2
from torchvision.transforms.v2 import InterpolationMode

__all__ = [
    "ALL_MODELS",
    "KnownModel",
    "ModelInternals",
    "load_model_internals",
    "lookup_model",
]


class _TorchvisionModel(Enum):
    ALEXNET = "alexnet"
    VGG_11 = "vgg11"
    VGG_13 = "vgg13"
    VGG_16 = "vgg16"
    VGG_19 = "vgg19"
    VGG_11_BN = "vgg11_bn"
    VGG_13_BN = "vgg13_bn"
    VGG_16_BN = "vgg16_bn"
    VGG_19_BN = "vgg19_bn"
    RESNET_18 = "resnet18"
    RESNET_34 = "resnet34"
    RESNET_50 = "resnet50"
    RESNET_101 = "resnet101"
    RESNET_152 = "resnet152"
    SQUEEZENET_1_0 = "squeezenet1_0"
    SQUEEZENET_1_1 = "squeezenet1_1"
    DENSENET_121 = "densenet121"
    DENSENET_161 = "densenet161"
    DENSENET_169 = "densenet169"
    DENSENET_201 = "densenet201"
    INCEPTION_V3 = "inception_v3"
    GOOGLENET = "googlenet"
    SHUFFLENET_V2_0_5 = "shufflenet_v2_x0_5"
    SHUFFLENET_V2_1_0 = "shufflenet_v2_x1_0"
    SHUFFLENET_V2_1_5 = "shufflenet_v2_x1_5"
    SHUFFLENET_V2_2_0 = "shufflenet_v2_x2_0"
    MOBILENET_V2 = "mobilenet_v2"
    MOBILENET_V3_L = "mobilenet_v3_large"
    MOBILENET_V3_S = "mobilenet_v3_small"
    RESNEXT_50 = "resnext50_32x4d"
    RESNEXT_101_32 = "resnext101_32x8d"
    RESNEXT_101_64 = "resnext101_64x4d"
    WIDERESNET_50_2 = "wide_resnet50_2"
    WIDERESNET_101_2 = "wide_resnet101_2"
    MNASNET_0_5 = "mnasnet0_5"
    MNASNET_0_75 = "mnasnet0_75"
    MNASNET_1_0 = "mnasnet1_0"
    MNASNET_1_3 = "mnasnet1_3"
    VIT_B_16 = "vit_b_16"
    VIT_B_32 = "vit_b_32"
    VIT_L_16 = "vit_l_16"
    VIT_L_32 = "vit_l_32"
    VIT_H_14 = "vit_h_14"
    SWIN_T = "swin_t"
    SWIN_S = "swin_s"
    SWIN_B = "swin_b"
    SWIN_V2_T = "swin_v2_t"
    SWIN_V2_S = "swin_v2_s"
    SWIN_V2_B = "swin_v2_b"
    CONVNEXT_TINY = "convnext_tiny"
    CONVNEXT_SMALL = "convnext_small"
    CONVNEXT_BASE = "convnext_base"
    CONVNEXT_LARGE = "convnext_large"
    EFFICIENTNET_B0 = "efficientnet_b0"
    EFFICIENTNET_B1 = "efficientnet_b1"
    EFFICIENTNET_B2 = "efficientnet_b2"
    EFFICIENTNET_B3 = "efficientnet_b3"
    EFFICIENTNET_B4 = "efficientnet_b4"
    EFFICIENTNET_B5 = "efficientnet_b5"
    EFFICIENTNET_B6 = "efficientnet_b6"
    EFFICIENTNET_B7 = "efficientnet_b7"
    REGNET_Y_400MF = "regnet_y_400mf"
    REGNET_Y_800MF = "regnet_y_800mf"
    REGNET_Y_1_6GF = "regnet_y_1_6gf"
    REGNET_Y_3_2GF = "regnet_y_3_2gf"
    REGNET_Y_8GF = "regnet_y_8gf"
    REGNET_Y_16GF = "regnet_y_16gf"
    REGNET_Y_32GF = "regnet_y_32gf"
    REGNET_Y_128GF = "regnet_y_128gf"
    REGNET_X_400MF = "regnet_x_400mf"
    REGNET_X_800MF = "regnet_x_800mf"
    REGNET_X_1_6GF = "regnet_x_1_6gf"
    REGNET_X_3_2GF = "regnet_x_3_2gf"
    REGNET_X_8GF = "regnet_x_8gf"
    REGNET_X_16GF = "regnet_x_16gf"
    REGNET_X_32GF = "regnet_x_32gf"
    MAXVIT_T = "maxvit_t"


class Transform:
    def __init__(
        self,
        crop_size: int = 224,
        resize_size: int = 256,
        mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        interpolation: tt2.InterpolationMode = tt2.InterpolationMode.BILINEAR,
        antialiasing: bool | None = True,
    ) -> None:
        self.crop_size = crop_size
        self.resize_size = resize_size
        self.mean = mean
        self.std = std
        self.interpolation = interpolation
        self.antialiasing = antialiasing

    def build(self) -> tt2.Compose:
        return tt2.Compose(
            [
                tt2.Resize(
                    (self.resize_size, self.resize_size),
                    self.interpolation,
                    antialias=self.antialiasing,
                ),
                tt2.CenterCrop(self.crop_size),
                tt2.Normalize(mean=self.mean, std=self.std),
            ]
        )


_MODEL_TRANSFORMATIONS = {
    "ALEXNET": Transform(),
    "VGG_11": Transform(),
    "VGG_13": Transform(),
    "VGG_16": Transform(),
    "VGG_19": Transform(),
    "VGG_11_BN": Transform(),
    "VGG_13_BN": Transform(),
    "VGG_16_BN": Transform(),
    "VGG_19_BN": Transform(),
    "RESNET_18": Transform(),
    "RESNET_34": Transform(),
    "RESNET_50": Transform(),
    "RESNET_101": Transform(),
    "RESNET_152": Transform(),
    "SQUEEZENET_1_0": Transform(),
    "SQUEEZENET_1_1": Transform(),
    "DENSENET_121": Transform(),
    "DENSENET_161": Transform(),
    "DENSENET_169": Transform(),
    "DENSENET_201": Transform(),
    "INCEPTION_V3": Transform(crop_size=299, resize_size=342),
    "GOOGLENET": Transform(),
    "SHUFFLENET_V2_0_5": Transform(),
    "SHUFFLENET_V2_1_0": Transform(),
    "SHUFFLENET_V2_1_5": Transform(),
    "SHUFFLENET_V2_2_0": Transform(),
    "MOBILENET_V2": Transform(),
    "MOBILENET_V3_L": Transform(),
    "MOBILENET_V3_S": Transform(),
    "RESNEXT_50": Transform(),
    "RESNEXT_101_32": Transform(),
    "RESNEXT_101_64": Transform(crop_size=224, resize_size=232),
    "WIDERESNET_50_2": Transform(),
    "WIDERESNET_101_2": Transform(),
    "MNASNET_0_5": Transform(),
    "MNASNET_0_75": Transform(),
    "MNASNET_1_0": Transform(),
    "MNASNET_1_3": Transform(resize_size=232),
    "VIT_B_16": Transform(),
    "VIT_B_32": Transform(),
    "VIT_L_16": Transform(resize_size=242),
    "VIT_L_32": Transform(),
    "VIT_H_14": Transform(
        resize_size=518, crop_size=518, interpolation=InterpolationMode.BICUBIC
    ),
    "SWIN_T": Transform(resize_size=232, interpolation=InterpolationMode.BICUBIC),
    "SWIN_S": Transform(resize_size=246, interpolation=InterpolationMode.BICUBIC),
    "SWIN_B": Transform(resize_size=238, interpolation=InterpolationMode.BICUBIC),
    "SWIN_V2_T": Transform(
        crop_size=256, resize_size=260, interpolation=InterpolationMode.BICUBIC
    ),
    "SWIN_V2_S": Transform(
        crop_size=256, resize_size=260, interpolation=InterpolationMode.BICUBIC
    ),
    "SWIN_V2_B": Transform(
        crop_size=256, resize_size=272, interpolation=InterpolationMode.BICUBIC
    ),
    "CONVNEXT_TINY": Transform(resize_size=236),
    "CONVNEXT_SMALL": Transform(resize_size=230),
    "CONVNEXT_BASE": Transform(resize_size=232),
    "CONVNEXT_LARGE": Transform(resize_size=232),
    "EFFICIENTNET_B0": Transform(interpolation=InterpolationMode.BICUBIC),
    "EFFICIENTNET_B1": Transform(
        crop_size=240, interpolation=InterpolationMode.BICUBIC
    ),
    "EFFICIENTNET_B2": Transform(
        resize_size=288, crop_size=288, interpolation=InterpolationMode.BICUBIC
    ),
    "EFFICIENTNET_B3": Transform(
        resize_size=320, crop_size=300, interpolation=InterpolationMode.BICUBIC
    ),
    "EFFICIENTNET_B4": Transform(
        resize_size=384, crop_size=380, interpolation=InterpolationMode.BICUBIC
    ),
    "EFFICIENTNET_B5": Transform(
        resize_size=456, crop_size=456, interpolation=InterpolationMode.BICUBIC
    ),
    "EFFICIENTNET_B6": Transform(
        resize_size=528, crop_size=528, interpolation=InterpolationMode.BICUBIC
    ),
    "EFFICIENTNET_B7": Transform(
        resize_size=600, crop_size=600, interpolation=InterpolationMode.BICUBIC
    ),
    "REGNET_Y_400MF": Transform(),
    "REGNET_Y_800MF": Transform(),
    "REGNET_Y_1_6GF": Transform(),
    "REGNET_Y_3_2GF": Transform(),
    "REGNET_Y_8GF": Transform(),
    "REGNET_Y_16GF": Transform(),
    "REGNET_Y_32GF": Transform(),
    "REGNET_Y_128GF": Transform(crop_size=384, resize_size=384),
    "REGNET_X_400MF": Transform(),
    "REGNET_X_800MF": Transform(),
    "REGNET_X_1_6GF": Transform(),
    "REGNET_X_3_2GF": Transform(),
    "REGNET_X_8GF": Transform(),
    "REGNET_X_16GF": Transform(),
    "REGNET_X_32GF": Transform(),
    "MAXVIT_T": Transform(resize_size=224, interpolation=InterpolationMode.BICUBIC),
    "REPVGG_A0": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_A1": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_A2": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B0": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B1": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B1G4": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B2": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B2G4": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B3": Transform(interpolation=InterpolationMode.BICUBIC),
    "REPVGG_B3G4": Transform(interpolation=InterpolationMode.BICUBIC),
    "MIXER_S32_224": Transform(),
    "MIXER_S16_224": Transform(),
    "MIXER_B32_224": Transform(),
    "MIXER_B16_224": Transform(),
    "MIXER_L32_224": Transform(),
    "MIXER_L16_224": Transform(),
    "POOLFORMER_S12": Transform(
        resize_size=248, interpolation=InterpolationMode.BICUBIC
    ),
    "POOLFORMER_S24": Transform(
        resize_size=248, interpolation=InterpolationMode.BICUBIC
    ),
    "POOLFORMER_S36": Transform(
        resize_size=248, interpolation=InterpolationMode.BICUBIC
    ),
    "POOLFORMER_M36": Transform(
        resize_size=236, interpolation=InterpolationMode.BICUBIC
    ),
    "POOLFORMER_M48": Transform(
        resize_size=236, interpolation=InterpolationMode.BICUBIC
    ),
}


class _OpenClipSpec(NamedTuple):
    model_name: str
    pretrained: str


class _OpenClipModel(Enum):
    MOBILECLIP_S1 = _OpenClipSpec("MobileCLIP-S1", "datacompdr")
    MOBILECLIP_B = _OpenClipSpec("MobileCLIP-B", "datacompdr")


class _TimmModel(Enum):
    REPVGG_A0 = "repvgg_a0"
    REPVGG_A1 = "repvgg_a1"
    REPVGG_A2 = "repvgg_a2"
    REPVGG_B0 = "repvgg_b0"
    REPVGG_B1 = "repvgg_b1"
    REPVGG_B1G4 = "repvgg_b1g4"
    REPVGG_B2 = "repvgg_b2"
    REPVGG_B2G4 = "repvgg_b2g4"
    REPVGG_B3 = "repvgg_b3"
    REPVGG_B3G4 = "repvgg_b3g4"
    MIXER_S32_224 = "mixer_s32_224"
    MIXER_S16_224 = "mixer_s16_224"
    MIXER_B32_224 = "mixer_b32_224"
    MIXER_B16_224 = "mixer_b16_224"
    MIXER_L32_224 = "mixer_l32_224"
    MIXER_L16_224 = "mixer_l16_224"
    POOLFORMER_S12 = "poolformer_s12"
    POOLFORMER_S24 = "poolformer_s24"
    POOLFORMER_S36 = "poolformer_s36"
    POOLFORMER_M36 = "poolformer_m36"
    POOLFORMER_M48 = "poolformer_m48"


KnownModel = _TorchvisionModel | _OpenClipModel | _TimmModel


@dataclass(frozen=True)
class ModelInternals:
    modules: list[tnn.Module]
    transform: tt2.Compose
    class_token: tnn.Parameter | None = None


def load_model_internals(model: KnownModel) -> ModelInternals:
    match model:
        case _TorchvisionModel():
            name = model.value
            weights = tvm.get_model_weights(name)["DEFAULT"]
            donor = getattr(tvm, name)(weights=weights)
            return ModelInternals(
                modules=list(donor.children()),
                transform=_MODEL_TRANSFORMATIONS[model.name.upper()].build(),
                class_token=donor.class_token if name.startswith("vit_") else None,
            )
        case _OpenClipModel():
            spec = model.value
            clip_model, _, preprocess = open_clip.create_model_and_transforms(
                spec.model_name, pretrained=spec.pretrained
            )
            return ModelInternals(
                modules=list(clip_model.visual.children())[:1],
                transform=preprocess,
            )
        case _TimmModel():
            name = model.value
            try:
                donor = timm.create_model(name, pretrained=True)
            except RuntimeError as err:
                if not str(err).startswith("No pretrained weights exist"):
                    raise
                donor = timm.create_model(name)
            return ModelInternals(
                modules=list(donor.children()),
                transform=_MODEL_TRANSFORMATIONS[model.name.upper()].build(),
            )
        case _:
            raise NotImplementedError(f"Unkown model: {model!r}")


ALL_MODELS: dict[str, KnownModel] = {
    model.name: model
    for cls in (_TorchvisionModel, _OpenClipModel, _TimmModel)
    for model in cls
}


def lookup_model(name: str) -> KnownModel:
    key = name.upper()
    if key not in ALL_MODELS:
        valid = ", ".join(sorted(ALL_MODELS))
        raise ValueError(f"Unknown model {name!r}. Valid names: {valid}")
    return ALL_MODELS[key]
