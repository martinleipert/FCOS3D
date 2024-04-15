import torch
from torch import nn


# Intersection Over Union Loss
# Part from https://github.com/rosinality/fcos-pytorch/blob/master/loss.py

# The function expects centered Boxes

class IOULoss(nn.Module):
    def __init__(self, loc_loss_type='iou'):
        super().__init__()

        self.loc_loss_type = loc_loss_type

    def forward(self, out, target, weight=None):
        pred_left, pred_top, pred_front, pred_right, pred_bottom, pred_back = out.unbind(1)
        target_left, target_top, target_front, target_right, target_bottom, target_back = target.unbind(1)

        target_area = (target_left + target_right) * (target_top + target_bottom) * (target_front + target_back)
        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom) * (pred_front + pred_back)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)

        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)

        d_intersect = torch.min(pred_front, target_front) + torch.min(pred_back, target_back)

        area_intersect = w_intersect * h_intersect * d_intersect
        area_union = target_area + pred_area - area_intersect

        ious = torch.divide(area_intersect + 1, area_union + 1)

        if self.loc_loss_type == 'iou':
            loss = -torch.log(ious)

        elif self.loc_loss_type == 'niou':
            loss = 1 - ious

        elif self.loc_loss_type == 'giou':
            g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
            g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
            g_d_intersect = torch.max(pred_front, target_front) + torch.max(pred_back, target_back)

            g_intersect = g_w_intersect * g_h_intersect * g_d_intersect + 1e-7
            gious = ious - (g_intersect - area_union) / g_intersect

            loss = 1 - gious

        if weight is not None and weight.sum() > 0:
            return torch.multiply(loss, weight).sum() / weight.sum()

        else:
            return loss.mean()
