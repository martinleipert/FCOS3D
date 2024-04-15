import torch
from torch.nn import Module

from fcos3d.Subnets.Backbone import ResSEBackbone
from fcos3d.Subnets.FCOSHead import FCOSHead
from fcos3d.Subnets.FPN import ResSEFPN
from fcos3d.Subnets.FCOSPostprocessor import FCOSPostprocessor
from fcos3d.Losses.FCOSLoss import FCOSLoss

"""
Martin Leipert
martin.leipert@fau.de
17.03.2023

Light version of CenterMask for 3D Computed Tomography Data
=> Lightweight Backends like Residual SE UNet
=> Implementation for 3D

Based on the Implementation by rosinality: 
https://github.com/rosinality/fcos-pytorch
"""


class FCOS3D(Module):

    def __init__(self,
                 # Number of input channels
                 in_channels=1,
                 # Number of output classes
                 n_classes=3,
                 # Feature maps of the backbone at level 0
                 f_maps=32,
                 # Number of levels in the backbone => downscale 2^backbone_levels
                 backbone_levels=5,
                 # Number of backbone levels that forward into FPN
                 backbone_out_levels=3,
                 # additional levels in fpn additionally to last backbone layer
                 fpn_add_levels=2,
                 # Number of groups in GroupNorm and grouped Convolutions
                 num_groups=8,
                 # Size of Input
                 image_size=(256, 256, 256),
                 # Number of cnvolutions in the fcos head
                 n_conv_fcos=3,
                 fcos_prior=0.5,
                 post_threshold=0.25,
                 # Number of proposals selected with best score
                 thresh_top_n=1000,
                 # Threshold for non maximum supression
                 nms_threshold=0.2,
                 # After nms number of top proposals
                 post_top_n=100,
                 # Minimal box size of proposal boxes
                 post_min_size=5,
                 # Loss paramters
                 # Focal Loss Gamma => Soft / Hard focal Loss
                 focal_loss_gamma=2,
                 # Alpha weigthing of foreground
                 focal_loss_alpha=0.95,
                 # Loss type for Boxes intersection
                 iou_loss_type='iou',
                 center_sample=True,
                 pos_radius=1.25,
                 # Handle box overlap exactly by next center
                 advanced_overlap_handling=False,
                 # Use center of mass for exact ground truth
                 use_center_of_mass=True,
                 # Use exact mask for loss assignment
                 use_exact_mask=True,
                 # Device
                 device="cuda"):

        super(FCOS3D, self).__init__()

        self.image_size = torch.Tensor(image_size)

        # Calculate the scale levels of the backbone
        first_level = backbone_levels - backbone_out_levels + 1
        last_level = backbone_levels + backbone_out_levels

        self.fpn_strides = torch.Tensor([pow(2, index) for index in range(first_level, last_level)])
        self.sizes = list()

        self.sizes_of_interest = list()

        minimal_size = -1
        maximal_size = max(image_size)

        for index, stride in enumerate(self.fpn_strides):
            self.sizes.append(torch.divide(self.image_size, stride))

            if index < (last_level - 1) and index > first_level:
                level_max_size = 8 * stride
                self.sizes_of_interest.append(torch.Tensor((minimal_size, level_max_size)).to(device))
                minimal_size = level_max_size
            else:
                self.sizes_of_interest.append(torch.Tensor((minimal_size, maximal_size)).to(device))

        self.scale_sizes = 1. / self.fpn_strides

        self.n_classes = n_classes

        self.add_module("backbone", ResSEBackbone(f_maps, in_channels, backbone_levels, out_levels=backbone_out_levels))

        f_maps_inout = f_maps * pow(2, backbone_levels - backbone_out_levels + 1)

        self.add_module("fpn", ResSEFPN(f_maps_inout, levels=backbone_out_levels, add_levels=fpn_add_levels,
                                        num_groups=num_groups))

        self.FCOS_head = FCOSHead(f_maps_inout, n_classes, n_conv_fcos, fcos_prior)

        self.FCOS_Postprocessor = FCOSPostprocessor(post_threshold, thresh_top_n, nms_threshold, post_top_n,
                                                    post_min_size, n_classes, self.image_size)

        self.fcos_loss = FCOSLoss(self.sizes_of_interest, focal_loss_gamma, focal_loss_alpha, iou_loss_type,
                                  center_sample, self.fpn_strides, pos_radius, self.image_size,
                                  advanced_overlap_handling, use_center_of_mass, use_exact_mask)

    def forward(self, x, targets=None):

        # x_in = x

        # Initial shape is image size

        x, x_bb = self.backbone(x)

        x_fpn = self.fpn(x, x_bb)

        logits, bboxes, centers = self.FCOS_head(x_fpn)

        location = self.compute_location(x_fpn)

        # strides = [8, 16, 32, 64, 128]
        # bboxes_post = []

        # for stride, boxes in zip(strides, bboxes):
        #    bboxes_post.append(stride*boxes)

        if self.training is False:
            predicted_boxes = self.FCOS_Postprocessor(location, logits, bboxes, centers, self.sizes)
            return predicted_boxes
        # predicted_boxes = self.FCOS_Postprocessor(location, logits, bboxes, centers, self.sizes)

        # target_boxes = targets["boxes"]
        # target_labels = targets["labels"]
        else:
            fc_loss = self.fcos_loss(location, logits, bboxes, centers, targets)
            return fc_loss
        # instances = Instances((128, 128, 128))
        # instances.set("pred_boxes", predicted_boxes)
        # instances.set("proposal_boxes", predicted_boxes)

        #  output = self.roi_pooler(x_fpn, predicted_boxes)

        # return predicted_boxes, fc_loss

    def compute_location(self, features):
        locations = []

        for i, feat in enumerate(features):
            _, _, height, width, depth = feat.shape
            location_per_level = self.compute_location_per_level(
                height, width, depth, self.fpn_strides[i], feat.device
            )
            locations.append(location_per_level)

        # locations = list(reversed(locations))

        return locations

    def compute_location_per_level(self, height, width, depth, stride, device):
        shift_x = torch.arange(
            0, width * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y = torch.arange(
            0, height * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_z = torch.arange(
            0, depth * stride, step=stride, dtype=torch.float32, device=device
        )
        shift_y, shift_x, shift_z = torch.meshgrid(shift_y, shift_x, shift_z)
        shift_x = shift_x.reshape(-1)
        shift_y = shift_y.reshape(-1)
        shift_z = shift_z.reshape(-1)
        location = torch.stack((shift_x, shift_y, shift_z), 1) + stride // 2

        return location
