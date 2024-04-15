import math
import torch
from torch import nn
from torch.nn import ModuleList, Module, Sequential, GroupNorm, ReLU, Conv3d
from fcos3d.Utils.Utility import init_conv_std
from fcos3d.Subnets.Blocks.Scale import Scale


class FCOSHead(Module):
    def __init__(self, in_channel, n_class, n_conv, prior, scales=None, groups=32):
        super().__init__()

        n_class = n_class

        cls_tower = []
        bbox_tower = []

        for i in range(n_conv):
            # Center and class path
            cls_tower.append(
                Conv3d(in_channel, in_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
            )
            cls_tower.append(GroupNorm(groups, in_channel))
            cls_tower.append(ReLU())

            # BBox path
            bbox_tower.append(
                Conv3d(in_channel, in_channel, kernel_size=(3, 3, 3), padding=(1, 1, 1), bias=False)
            )
            bbox_tower.append(GroupNorm(groups, in_channel))
            bbox_tower.append(ReLU())
            pass

        self.cls_tower = Sequential(*cls_tower)
        self.bbox_tower = Sequential(*bbox_tower)

        # Prediction heads
        # Cls pred and center share the same branch
        self.cls_pred = Conv3d(in_channel, n_class, kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.center_pred = Conv3d(in_channel, 1, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        # BBox predictions based on separate branch
        self.bbox_pred = Conv3d(in_channel, 6, kernel_size=(3, 3, 3), padding=(1, 1, 1))

        self.apply(init_conv_std)

        prior_bias = -math.log((1 - prior) / prior)
        nn.init.constant_(self.cls_pred.bias, prior_bias)

        if scales is None:
            self.scales = ModuleList([Scale(1.0) for _ in range(5)])
        else:
            self.scales = ModuleList([Scale(scale) for scale in scales])

    def forward(self, x):
        logits = []
        bboxes = []
        centers = []

        for feat, scale in zip(x, self.scales):
            # Class and center path
            cls_out = self.cls_tower(feat)

            logits.append(self.cls_pred(cls_out))
            centers.append(self.center_pred(cls_out))

            # BBox path
            bbox_out = self.bbox_tower(feat)
            bbox_out = torch.exp(scale(self.bbox_pred(bbox_out)))

            bboxes.append(bbox_out)

        return logits, bboxes, centers

