# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.models.backbones import SSDVGG, HRNet, ResNet, ResNetV1d, ResNeXt
from .resnet import CustomResNet, CustomResNet3D

__all__ = [
    'ResNet', 'ResNetV1d', 'ResNeXt', 'SSDVGG', 'HRNet', 'CustomResNet', 'CustomResNet3D',
]
