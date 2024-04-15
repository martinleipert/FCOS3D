import numpy
import torch
from torch import nn
from fcos3d.Losses.IOULosses import IOULoss
from fcos3d.Losses.SigmoidFocalLoss import SigmoidFocalLoss
from torch.nn.modules.loss import SmoothL1Loss

# Source:
# https://github.com/rosinality/fcos-pytorch/blob/master/loss.py

# Defintion of INF 1000^3
INF = 1000000000


class FCOSLoss(nn.Module):
    def __init__(
            self, sizes, gamma, alpha, iou_loss_type, center_sample, fpn_strides, pos_radius, img_full_size,
            advanced_overlap_handling=False, use_center_of_mass=False, use_exact_mask=False, l1_loss=False
    ):
        super().__init__()

        self.sizes = sizes

        self.cls_loss = SigmoidFocalLoss(gamma, alpha)  # # SigmoidFocalLoss(gamma, alpha) # #

        self.l1_loss = l1_loss

        if l1_loss is True:
            self.box_loss = SmoothL1Loss(reduce="mean")
        else:
            self.box_loss = IOULoss(iou_loss_type)  # # IOULoss(iou_loss_type)

        self.center_loss = nn.BCEWithLogitsLoss()

        self.center_sample = center_sample
        self.strides = fpn_strides
        self.radius = pos_radius

        self.img_full_size = img_full_size

        # Use center of mass to determine bbox
        self.advanced_overlap_handling = advanced_overlap_handling
        # Use center of mass for centerness computation
        self.use_center_of_mass = use_center_of_mass
        # Use the exact mask for class labeling
        self.use_exact_mask = use_exact_mask

    def prepare_target(self, points, targets):
        ex_size_of_interest = []

        for i, point_per_level in enumerate(points):
            # size_of_interest_per_level = point_per_level.new_tensor(self.sizes[i])
            size_of_interest_per_level = self.sizes[i].clone().detach()
            ex_size_of_interest.append(
                size_of_interest_per_level.expand(len(point_per_level), -1)
            )

        ex_size_of_interest = torch.cat(ex_size_of_interest, 0)
        n_point_per_level = [len(point_per_level) for point_per_level in points]
        point_all = torch.cat(points, dim=0)
        label, box_target, centers = self.compute_target_for_location(
            point_all, targets, ex_size_of_interest, n_point_per_level
        )

        for i in range(len(label)):
            label[i] = torch.split(label[i], n_point_per_level, 0)
            box_target[i] = torch.split(box_target[i], n_point_per_level, 0)

        label_level_first = []
        box_target_level_first = []

        for level in range(len(points)):
            label_level_first.append(
                torch.cat([label_per_img[level] for label_per_img in label], 0)
            )
            box_target_level_first.append(
                torch.cat(
                    [box_target_per_img[level] for box_target_per_img in box_target], 0
                )
            )

        return label_level_first, box_target_level_first, centers

    def get_sample_region(self, gt, strides, n_point_per_level, xs, ys, zs, radius=1):

        n_gt = gt.shape[0]
        n_loc = len(xs)
        gt = gt[None].expand(n_loc, n_gt, 6)

        # claculate the bounding box centers for each voxel
        center_x = (gt[..., 0] + gt[..., 3]) / 2
        center_y = (gt[..., 1] + gt[..., 4]) / 2
        center_z = (gt[..., 2] + gt[..., 5]) / 2

        if center_x[..., 0].sum() == 0:
            return xs.new_zeros(xs.shape, dtype=torch.uint8)

        begin = 0

        center_gt = gt.new_zeros(gt.shape)
        # Locate bounding box
        for level, n_p in enumerate(n_point_per_level):
            end = begin + n_p
            stride = strides[level] * radius

            x_min = center_x[begin:end] - stride
            y_min = center_y[begin:end] - stride
            z_min = center_z[begin:end] - stride
            x_max = center_x[begin:end] + stride
            y_max = center_y[begin:end] + stride
            z_max = center_z[begin:end] + stride

            center_gt[begin:end, :, 0] = torch.where(
                x_min > gt[begin:end, :, 0], x_min, gt[begin:end, :, 0]
            )
            center_gt[begin:end, :, 1] = torch.where(
                y_min > gt[begin:end, :, 1], y_min, gt[begin:end, :, 1]
            )
            center_gt[begin:end, :, 2] = torch.where(
                z_min > gt[begin:end, :, 2], z_min, gt[begin:end, :, 2]
            )
            center_gt[begin:end, :, 3] = torch.where(
                x_max > gt[begin:end, :, 3], gt[begin:end, :, 3], x_max
            )
            center_gt[begin:end, :, 4] = torch.where(
                y_max > gt[begin:end, :, 4], gt[begin:end, :, 4], y_max
            )
            center_gt[begin:end, :, 5] = torch.where(
                z_max > gt[begin:end, :, 5], gt[begin:end, :, 5], z_max
            )

            begin = end
        # Calculate Distance to bbox margin
        left = xs[:, None] - center_gt[..., 0]
        right = center_gt[..., 3] - xs[:, None]
        top = ys[:, None] - center_gt[..., 1]
        bottom = center_gt[..., 4] - ys[:, None]
        front = zs[:, None] - center_gt[..., 2]
        back = center_gt[..., 5] - zs[:, None]

        center_bbox = torch.stack((left, top, front, right, bottom, back), -1)
        is_in_boxes = center_bbox.min(-1)[0] > 0

        return is_in_boxes

    def get_sample_region_mask(self, mask_targets, n_point_per_level, locations):
        device = mask_targets.device
        shape = mask_targets.shape[1:]

        indices = torch.Tensor([shape[1] * shape[2], shape[2], 1]).to(device) * locations
        indices = indices.long().sum(1)

        is_in_boxes = torch.full((indices.shape[0], mask_targets.shape[0]), False, device=device)

        for idx in range(mask_targets.shape[0]):
            box_values = mask_targets[idx].flatten()[indices]
            is_in_boxes[:, idx] = box_values
        return is_in_boxes

    def compute_target_for_location(
            self, locations, targets, sizes_of_interest, n_point_per_level, mask_targets=None
    ):
        labels = []
        box_targets = []
        centers = []

        # Target coordinates element wise
        xs, ys, zs = locations[:, 0], locations[:, 1], locations[:, 2]

        mask_targets = targets.get_field("masks")
        centers_of_mass = targets.get_field("centers")

        # Iterate over the batch => targets for single image
        for i in range(len(targets)):
            centers_per_img = centers_of_mass[i]
            targets_per_img = targets[i]
            assert targets_per_img.mode == 'xyzxyz'
            bboxes = targets_per_img.box
            labels_per_img = targets_per_img.extra_fields['labels']
            area = targets_per_img.area()

            # Box targets
            # Relative to the real box centers
            l = xs[:, None] - bboxes[:, 0][None]
            t = ys[:, None] - bboxes[:, 1][None]
            f = zs[:, None] - bboxes[:, 2][None]
            r = bboxes[:, 3][None] - xs[:, None]
            b = bboxes[:, 4][None] - ys[:, None]
            a = bboxes[:, 5][None] - zs[:, None]

            # Centered Boxes for each voxel
            # Stack at third axis
            box_targets_per_img = torch.stack([l, t, f, r, b, a], 2)

            # Check if the voxels lie in boxes
            if self.center_sample:
                is_in_boxes = self.get_sample_region(
                    bboxes, self.strides, n_point_per_level, xs, ys, zs, radius=self.radius
                )
            else:
                is_in_boxes = box_targets_per_img.min(2)[0] > 0

            if self.use_exact_mask:
                is_in_boxes_new = self.get_sample_region_mask(
                    mask_targets[i], n_point_per_level, locations
                )
                is_in_boxes = torch.logical_and(is_in_boxes_new, is_in_boxes)

            # START NEW
            # Find overlapping boxes
            if self.advanced_overlap_handling and not self.use_exact_mask:
                number_of_boxes = torch.sum(is_in_boxes, 1)
                overlapping = number_of_boxes > 1
                overlapping_indices = torch.argwhere(overlapping).squeeze(1)

                overlap_mask = is_in_boxes[overlapping_indices]
                overlap_locations = locations[overlapping_indices].unsqueeze(1)
                overlap_centers = torch.tile(centers_of_mass, (overlapping_indices.size(0), 1, 1))

                distances = torch.subtract(overlap_locations, overlap_centers)
                distances = torch.norm(distances, 'fro', 2)
                min_indices = torch.argmin(distances, 1)

                is_in_boxes[overlapping_indices] = False
                is_in_boxes[overlapping_indices, min_indices] = True

                # print(torch.sum(is_in_boxes))

            # END NEW

            max_box_targets_per_img = box_targets_per_img.max(2)[0]

            # Is cared in level saves
            # Is the box relevant for this level?
            is_cared_in_level = (
                                        max_box_targets_per_img >= sizes_of_interest[:, [0]]
                                ) & (
                                        max_box_targets_per_img <= sizes_of_interest[:, [1]]
                                )

            locations_to_gt_area = area[None].repeat(len(locations), 1)
            locations_to_gt_area[is_in_boxes == 0] = INF
            locations_to_gt_area[is_cared_in_level == 0] = INF

            #
            # Location => Default value is extremely high, therefore minimum is 0
            locations_to_min_area, locations_to_gt_id = locations_to_gt_area.min(1)

            box_targets_per_img = box_targets_per_img[
                range(len(locations)), locations_to_gt_id
            ]

            labels_per_img = torch.Tensor(labels_per_img)[locations_to_gt_id]
            labels_per_img[locations_to_min_area == INF] = 0

            img_centers = centers_per_img[locations_to_gt_id]

            centers.append(img_centers)
            labels.append(labels_per_img.detach())
            box_targets.append(box_targets_per_img)

        return labels, box_targets, centers

    def compute_centerness_targets(self, box_targets):

        left_right = box_targets[:, [0, 3]]
        top_bottom = box_targets[:, [1, 4]]
        front_back = box_targets[:, [2, 5]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * \
                     (top_bottom.min(-1)[0] / top_bottom.max(-1)[0]) * \
                     (front_back.min(-1)[0] / front_back.max(-1)[0])

        centerness = torch.clip(centerness, 1e-8, 1)
        return torch.sqrt(centerness)

    def compute_centerness_target_new(self, centers_of_mass, locations, bboxes):

        lower_bound = torch.subtract(locations, bboxes[:, 0:3])
        upper_bound = torch.add(locations, bboxes[:, 3:6])

        # COnvert to bbox in previous styles
        distance_to_center_lower = torch.subtract(centers_of_mass, lower_bound)
        distance_to_center_upper = torch.subtract(upper_bound, centers_of_mass)

        # Get the location relative to center of mass
        relative_location = torch.subtract(locations, centers_of_mass)

        # Check if above or below zero => which side of center is important
        upper_or_lower = relative_location > 0
        divisor = torch.where(upper_or_lower, distance_to_center_upper, distance_to_center_lower)

        # Relative position
        relative_location = torch.divide(relative_location, divisor + 1e-8)

        box_targets = torch.hstack((torch.add(1, relative_location), torch.subtract(1, relative_location)))

        left_right = box_targets[:, [0, 3]]
        top_bottom = box_targets[:, [1, 4]]
        front_back = box_targets[:, [2, 5]]
        centerness = (left_right.min(-1)[0] / left_right.max(-1)[0]) * \
                     (top_bottom.min(-1)[0] / top_bottom.max(-1)[0]) * \
                     (front_back.min(-1)[0] / front_back.max(-1)[0])

        # Now calculate for normalization
        centerness = torch.clip(centerness, 1e-8, 1)
        return torch.sqrt(centerness)

    def forward(self, locations, cls_pred, box_pred, center_pred, targets):
        batch = cls_pred[0].shape[0]
        n_class = cls_pred[0].shape[1]

        labels, box_targets, centers = self.prepare_target(locations, targets)

        cls_flat = []
        box_flat = []
        center_flat = []

        labels_flat = []
        box_targets_flat = []

        for i in range(len(labels)):
            cls_flat.append(cls_pred[i].permute(0, 2, 3, 4, 1).reshape(-1, n_class))
            box_flat.append(box_pred[i].permute(0, 2, 3, 4, 1).reshape(-1, 6))
            center_flat.append(center_pred[i].permute(0, 2, 3, 4, 1).reshape(-1))

            labels_flat.append(labels[i].reshape(-1))
            box_targets_flat.append(box_targets[i].reshape(-1, 6))

        cls_flat = torch.cat(cls_flat, 0)
        box_flat = torch.cat(box_flat, 0)
        center_flat = torch.cat(center_flat, 0)
        com_flat = torch.cat(centers)

        labels_flat = torch.cat(labels_flat, 0).to(torch.int32)
        box_targets_flat = torch.cat(box_targets_flat, 0)

        pos_id = torch.nonzero(labels_flat > 0).squeeze(1)
        cls_loss = self.cls_loss(cls_flat, labels_flat.int()) / (pos_id.numel() + batch)

        box_flat = box_flat[pos_id]
        center_flat = center_flat[pos_id]
        com_flat = com_flat[pos_id]
        locations_flat = torch.cat(locations, 0)[pos_id]

        box_targets_flat = box_targets_flat[pos_id]

        if pos_id.numel() > 0:

            if not self.use_center_of_mass:
                center_targets = self.compute_centerness_targets(box_targets_flat)
            else:
                center_targets = self.compute_centerness_target_new(com_flat, locations_flat, box_targets_flat)

            if self.l1_loss is True:
                box_loss = self.box_loss(box_flat, box_targets_flat)
            else:
                box_loss = self.box_loss(box_flat, box_targets_flat, center_targets)
            center_loss = self.center_loss(center_flat, center_targets)

        else:
            box_loss = box_flat.sum()
            center_loss = center_flat.sum()

        return cls_loss, box_loss, center_loss
