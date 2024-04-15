import torch
from torchvision import ops

from fcos3d.Utils.nms import batched_nms, ml_nms, class_nms

FLIP_LEFT_RIGHT = 0
FLIP_TOP_BOTTOM = 1


class BoxList:
    def __init__(self, box, image_size, mode='xyzxyz', image_full_size=[128, 128, 128]):
        device = box.device if hasattr(box, 'device') else 'cpu'
        box = torch.as_tensor(box, dtype=torch.float32, device=device)

        self.box = box
        self.size = image_size
        self.mode = mode
        self.image_full_size = image_full_size

        self.extra_fields = {}

    def convert(self, mode):
        if mode == self.mode:
            return self

        x_min, y_min, z_min, x_max, y_max, z_max = self.split_to_xyzxyz()

        if mode == 'xyzxyz':
            box = torch.cat([x_min, y_min, z_min, x_max, y_max, z_max], -1)
            box = BoxList(box, self.size, mode=mode)

        elif mode == 'xyzwhd':
            remove = 1
            box = torch.cat(
                [x_min, y_min, z_min, x_max - x_min + remove, y_max - y_min + remove, z_max - z_min + remove], -1
            )
            box = BoxList(box, self.size, mode=mode)

        box.copy_field(self)

        return box

    def add_field(self, field, field_data):
        self.extra_fields[field] = field_data

    def get_field(self, field):
        return self.extra_fields[field]

    def has_field(self, field):
        return field in self.extra_fields

    def extra_fields(self):
        return list(self.extra_fields.keys())

    def copy_field(self, box):
        for k, v in box.extra_fields.items():
            self.extra_fields[k] = v

    def area(self):
        box = self.box

        if box.shape[1] == 0:
            return torch.zeros((box.shape[1],))

        if self.mode == 'xyzxyz':
            remove = 1

            area = (box[:, 3] - box[:, 0] + remove) * (box[:, 4] - box[:, 1] + remove) * \
                   (box[:, 5] - box[:, 2] + remove)

        elif self.mode == 'xyzwhd':
            area = box[:, 3] * box[:, 4] * box[:, 5]

        return area

    def split_to_xyzxyz(self):
        if self.mode == 'xyzxyz':
            x_min, y_min, z_min, x_max, y_max, z_max = self.box.split(1, dim=-1)

            return x_min, y_min, z_min, x_max, y_max, z_max

        elif self.mode == 'xyzwhd':
            remove = 1
            x_min, y_min, z_min, w, h, d = self.box.split(1, dim=-1)

            return (
                x_min,
                y_min,
                z_min,
                x_min + (w - remove).clamp(min=0),
                y_min + (h - remove).clamp(min=0),
                z_min + (d - remove).clamp(min=0)
            )

    def __len__(self):
        return self.box.shape[0]

    def __getitem__(self, index):
        box = BoxList(self.box[index], self.size, self.mode)

        for k, v in self.extra_fields.items():
            box.extra_fields[k] = v[index]

        return box

    def resize(self, size, *args, **kwargs):
        ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(size, self.size))

        if ratios[0] == ratios[1]:
            ratio = ratios[0]
            scaled = self.box * ratio
            box = BoxList(scaled, size, mode=self.mode)

            for k, v in self.extra_fields.items():
                if not isinstance(v, torch.Tensor):
                    v = v.resize(size, *args, **kwargs)

                box.extra_fields[k] = v

            return box

        ratio_w, ratio_h, ratio_d = ratios
        x_min, y_min, z_min, x_max, y_max, z_max = self.split_to_xyzxyz()
        scaled_x_min = x_min * ratio_w
        scaled_x_max = x_max * ratio_w
        scaled_y_min = y_min * ratio_h
        scaled_y_max = y_max * ratio_h
        scaled_z_min = z_min * ratio_d
        scaled_z_max = z_max * ratio_d
        scaled = torch.cat([scaled_x_min, scaled_y_min, scaled_z_min, scaled_x_max, scaled_y_max, scaled_z_max], -1)
        box = BoxList(scaled, size, mode='xyzxyz')

        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.resize(size, *args, **kwargs)

            box.extra_fields[k] = v

        return box.convert(self.mode)

    def transpose(self, method):
        width, height, depth = self.size
        x_min, y_min, z_min, x_max, y_max, z_max = self.split_to_xyzxyz()

        if method == FLIP_LEFT_RIGHT:
            remove = 1

            transpose_x_min = width - x_max - remove
            transpose_x_max = width - x_min - remove
            transpose_y_min = y_min
            transpose_y_max = y_max
            transpose_z_min = z_min
            transpose_z_max = z_max

        elif method == FLIP_TOP_BOTTOM:
            transpose_x_min = x_min
            transpose_x_max = x_max
            transpose_y_min = height - y_max - remove
            transpose_y_max = height - y_min - remove
            transpose_z_min = z_min
            transpose_z_max = z_max

        elif method == FLIP_FRONT_BACK:
            transpose_x_min = x_min
            transpose_x_max = x_max
            transpose_y_min = y_min
            transpose_y_max = y_max
            transpose_z_min = depth - z_max - remove
            transpose_z_max = depth - z_min - remove

        transpose_box = torch.cat(
            [transpose_x_min, transpose_y_min, transpose_z_min, transpose_x_max, transpose_y_max, transpose_z_max], -1
        )
        box = BoxList(transpose_box, self.size, mode='xyzxyz')

        for k, v in self.extra_fields.items():
            if not isinstance(v, torch.Tensor):
                v = v.transpose(method)

            box.extra_fields[k] = v

        return box.convert(self.mode)

    def clip(self, remove_empty=True):
        remove = 1

        max_width = self.size[0] - remove
        max_height = self.size[1] - remove
        max_depth = self.size[2] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_depth)
        self.box[:, 3].clamp_(min=0, max=max_width)
        self.box[:, 4].clamp_(min=0, max=max_height)
        self.box[:, 5].clamp_(min=0, max=max_depth)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 0]) & (box[:, 4] > box[:, 1]) & (box[:, 5] > box[:, 2])

            return self[keep]

        else:
            return self

    def clip_to_image(self, remove_empty=True):
        remove = 1

        max_width = self.image_full_size[0] - remove
        max_height = self.image_full_size[1] - remove
        max_depth = self.image_full_size[2] - remove

        self.box[:, 0].clamp_(min=0, max=max_width)
        self.box[:, 1].clamp_(min=0, max=max_height)
        self.box[:, 2].clamp_(min=0, max=max_depth)
        self.box[:, 3].clamp_(min=0, max=max_width)
        self.box[:, 4].clamp_(min=0, max=max_height)
        self.box[:, 5].clamp_(min=0, max=max_depth)

        if remove_empty:
            box = self.box
            keep = (box[:, 3] > box[:, 0]) & (box[:, 4] > box[:, 1]) & (box[:, 5] > box[:, 2])
            return self[keep]
        else:
            return self

    def to(self, device):
        box = BoxList(self.box.to(device), self.size, self.mode)

        for k, v in self.extra_fields.items():
            if hasattr(v, 'to'):
                v = v.to(device)

            box.extra_fields[k] = v

        return box


def remove_small_boxes(boxlist, min_size):
    box = boxlist.convert('xyzwhd').box
    _, _, _, w, h, d = box.unbind(dim=1)
    keep = (w >= min_size) & (h >= min_size) & (d >= min_size)
    keep = keep.nonzero().squeeze(1)

    return boxlist[keep]


def cat_boxlist(boxlists):
    size = boxlists[0].size
    mode = boxlists[0].mode
    field_keys = boxlists[0].extra_fields.keys()

    boxlists = list(filter(lambda x: x.box.shape[0] > 0, boxlists))
    if len(boxlists) == 0:
        return boxlists

    box_cat = torch.cat([boxlist.box for boxlist in boxlists], 0)
    new_boxlist = BoxList(box_cat, size, mode, boxlists[0].image_full_size)

    for field in field_keys:
        data = torch.cat([boxlist.extra_fields[field] for boxlist in boxlists], 0)
        new_boxlist.extra_fields[field] = data

    return new_boxlist


def boxlist_nms(boxlist, scores, threshold, max_proposal=-1):
    if threshold <= 0:
        return boxlist

    mode = boxlist.mode
    boxlist = boxlist.convert('xyzxyz')
    # nbox = boxlist.box
    keep = class_nms(boxlist.box, scores, threshold)

    if max_proposal > 0:
        keep = keep[:max_proposal]

    boxlist = boxlist[keep == 1]

    return boxlist.convert(mode)
