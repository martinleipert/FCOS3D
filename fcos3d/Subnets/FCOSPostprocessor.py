import torch
from torch import nn

"""
Martin Leipert
martin.leipert@th-deg.de

FCOS delivers the box for the segmentation center mask does, is closely linked to the backbone 

From the github Repo: 
https://github.com/rosinality/fcos-pytorch/blob/master/model.py 
"""

from fcos3d.Utils.Structures.BoxList import BoxList, boxlist_nms, cat_boxlist, remove_small_boxes


class FCOSPostprocessor(nn.Module):
    def __init__(self, threshold, top_n, nms_threshold, post_top_n, min_size, n_class, img_full_size):
        super().__init__()

        self.threshold = threshold
        self.top_n = top_n
        self.nms_threshold = nms_threshold
        self.post_top_n = post_top_n
        self.min_size = min_size
        self.n_class = n_class
        self.img_full_size = img_full_size

    def forward_single_feature_map(
            self, location, cls_pred, box_pred, center_pred, image_sizes
    ):
        batch, channel, height, width, depth = cls_pred.shape
        cls_pred = cls_pred.view(batch, channel, height, width, depth).permute(0, 2, 3, 4, 1)
        cls_pred = cls_pred.reshape(batch, -1, channel).sigmoid()

        box_pred = box_pred.view(batch, 6, height, width, depth).permute(0, 2, 3, 4, 1)
        box_pred = box_pred.reshape(batch, -1, 6)

        center_pred = center_pred.view(batch, 1, height, width, depth).permute(0, 2, 3, 4, 1)
        center_pred = center_pred.reshape(batch, -1).sigmoid()

        candid_ids = cls_pred > self.threshold
        # TODO check the changes here
        top_ns = candid_ids.reshape(batch, -1).sum(1)
        top_ns = top_ns.clamp(max=self.top_n)

        cls_pred = cls_pred * center_pred[:, :, None]

        results = []

        for i in range(batch):
            cls_p = cls_pred[i]
            candid_id = candid_ids[i]
            cls_p = cls_p[candid_id]
            candid_nonzero = candid_id.nonzero()
            box_loc = candid_nonzero[:, 0]
            class_id = candid_nonzero[:, 1] + 1

            box_p = box_pred[i]
            box_p = box_p[box_loc]
            loc = location[box_loc]

            top_n = top_ns[i]

            if candid_id.sum().item() > top_n.item():
                cls_p, top_k_id = cls_p.topk(top_n, sorted=True)
                class_id = class_id[top_k_id]
                box_p = box_p[top_k_id]
                loc = loc[top_k_id]

            detections = torch.stack(
                [
                    loc[:, 0] - box_p[:, 0],
                    loc[:, 1] - box_p[:, 1],
                    loc[:, 2] - box_p[:, 2],
                    loc[:, 0] + box_p[:, 3],
                    loc[:, 1] + box_p[:, 4],
                    loc[:, 2] + box_p[:, 5],
                ],
                1,
            )

            height, width, depth = self.img_full_size

            boxlist = BoxList(detections, (int(height), int(width), int(depth)), mode='xyzxyz')

            boxlist.extra_fields['labels'] = class_id
            boxlist.extra_fields['scores'] = torch.sqrt(cls_p)
            boxlist = boxlist.clip(remove_empty=False)
            boxlist = remove_small_boxes(boxlist, self.min_size)

            results.append(boxlist)

        return results

    def forward(self, location, cls_pred, box_pred, center_pred, image_sizes):
        boxes = []

        # strides = [8, 16, 32, 64, 128]

        for idx, (loc, cls_p, box_p, center_p) in enumerate(zip(
                location, cls_pred, box_pred, center_pred
        )):
            # box_p = box_p*strides[idx]
            boxes.append(
                self.forward_single_feature_map(
                    loc, cls_p.detach(), box_p.detach(), center_p.detach(), image_sizes
                )
            )

        boxlists = list(zip(*boxes))
        boxlists = [cat_boxlist(boxlist) for boxlist in boxlists]
        boxlists = self.select_over_scales(boxlists)

        return boxlists

    def select_over_scales(self, boxlists):
        results = []

        for boxlist in boxlists:
            if boxlist == []:
                continue

            scores = boxlist.extra_fields['scores']
            labels = boxlist.extra_fields['labels']
            box = boxlist.box

            result = []

            for j in range(1, self.n_class + 1):
                id = (labels == j).nonzero().view(-1)
                score_j = scores[id]
                box_j = box[id, :].view(-1, 6)
                box_by_class = BoxList(box_j, boxlist.size, mode='xyzxyz')
                box_by_class.extra_fields['scores'] = score_j.view(-1)
                box_by_class = boxlist_nms(box_by_class, score_j, self.nms_threshold)
                n_label = len(box_by_class)
                box_by_class.extra_fields['labels'] = torch.full(
                    (n_label,), j, dtype=torch.int64, device=scores.device
                )
                box_by_class.box = box_by_class.box.view(-1, 6)
                result.append(box_by_class)

            result = cat_boxlist(result)
            n_detection = len(result)

            if n_detection > self.post_top_n > 0:
                scores = result.extra_fields['scores']
                img_threshold, _ = torch.kthvalue(
                    scores.cpu(), n_detection - self.post_top_n + 1
                )
                keep = scores >= img_threshold.item()
                keep = torch.nonzero(keep).squeeze(1)
                result = result[keep]

            results.append(result)

        return results
