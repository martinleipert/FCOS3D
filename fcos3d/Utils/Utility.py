import torch
from torch import nn


def init_conv_kaiming(module):
    if isinstance(module, nn.Conv3d):
        nn.init.kaiming_uniform_(module.weight, a=1)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)


def init_conv_std(module, std=0.01):
    if isinstance(module, nn.Conv3d):
        nn.init.normal_(module.weight, std=std)

        if module.bias is not None:
            nn.init.constant_(module.bias, 0)

