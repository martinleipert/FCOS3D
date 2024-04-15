from pytorch3d.ops.iou_box3d import box3d_overlap
import torch
import numpy

INDICES_LIST = [
    [0, 1, 2],
    [3, 1, 2],
    [3, 4, 2],
    [0, 4, 2],
    [0, 1, 5],
    [3, 1, 5],
    [3, 4, 5],
    [0, 4, 5]
]


def class_nms(boxlist, scores, nms_thresh):
    boxes = boxlist

    if boxes.shape[0] == 0:
        return []

    shape = boxes.shape
    n_objects = shape[0]

    indices_list = torch.tensor(INDICES_LIST).flatten()
    boxes_reshaped = boxes[:, indices_list].reshape((n_objects, 8, 3))

    # Copy the tensor to get an iou_matrix
    boxes_reshaped = boxes_reshaped.detach().to("cpu")
    overlap_abs, iou = box3d_overlap(boxes_reshaped, boxes_reshaped)

    overlap_abs = overlap_abs.to(boxlist.device)
    iou = iou.to(boxlist.device)

    overlapping = iou > nms_thresh

    overlapping[torch.eye(n_objects) == 1] = False

    keep = torch.full_like(scores, False)

    if n_objects == 1:
        keep[0] == True
    else:

        score_array = torch.tile(scores.reshape([-1, 1]), [1, scores.shape[0]])
        intersect_scores = torch.where(overlapping, score_array, 0)


        for obj_index, score in enumerate(scores.detach().cpu().numpy()):
            obj_intersections = overlapping[:, obj_index]
            overlap_scores = intersect_scores[:, obj_index]

            max_score = torch.max(overlap_scores)
            if max_score > score:
                continue

            keep[obj_index] = 1

            # Surpress other boxes
            surpres_indices = torch.nonzero(overlap_scores)
            overlapping[surpres_indices, :] = False
            intersect_scores[surpres_indices, :] = 0

    return keep


def ml_nms(boxlist, nms_thresh, max_proposals=-1, score_field="scores", label_field="labels"):
    """
    Performs non-maximum suppression on a boxlist, with scores specified
    in a boxlist field via score_field.
    Arguments:
        boxlist(BoxList)
        nms_thresh (float)
        max_proposals (int): if > 0, then only the top max_proposals are kept
            after non-maximum suppression
        score_field (str)
    """
    if nms_thresh <= 0:
        return boxlist
    boxes = boxlist.pred_boxes.tensor
    scores = boxlist.scores
    labels = boxlist.pred_classes
    keep = batched_nms(boxes, scores, labels, nms_thresh)

    if max_proposals > 0:
        keep = keep[: max_proposals]
    boxlist = boxlist[keep]
    return boxlist


def batched_nms(boxes, scores, labels, nms_thresh):
    n_objects = labels.shape[0]

    shape = boxes.shape
    indices_list = torch.tensor(INDICES_LIST).flatten()
    boxes_reshaped = boxes[:, indices_list].reshape((shape[0], 8, 3))

    unique_labels = torch.unique(labels).to(torch.int32)

    # Copy the tensor to get an iou_matrix
    overlap_abs, iou = box3d_overlap(boxes_reshaped, boxes_reshaped)

    overlapping = iou > nms_thresh

    overlapping[range(n_objects), range(n_objects)] = False

    keep = torch.full_like(labels, False)

    for class_index in unique_labels:
        class_objects = labels == class_index
        num_objects = torch.count_nonzero(class_objects)

        if num_objects == 1:
            obj_index = torch.argwhere(class_objects)
            obj_index = obj_index[0, 0]
            keep[obj_index] = True
            continue

        class_object_indices = numpy.argwhere(class_objects)

        class_scores = scores[class_objects]

        class_intersections = overlapping[:, class_objects][class_objects, :]

        for obj_index, score in enumerate(class_scores.numpy()):

            keep_index = class_object_indices[0, obj_index]

            obj_intersections = class_intersections[:, obj_index]

            intersect_scores = class_scores[obj_intersections]

            if intersect_scores.shape[0] == 0:
                keep[keep_index] = True
            elif torch.all(score > intersect_scores):
                keep[keep_index] = True

    return keep
