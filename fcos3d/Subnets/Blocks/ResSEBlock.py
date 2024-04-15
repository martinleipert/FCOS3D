from torch import mul, cat
from torch.nn import Module, Linear, Sigmoid, AdaptiveAvgPool3d, ReLU, BatchNorm3d, MaxPool3d, ConvTranspose3d, Conv3d,\
    Sequential, LeakyReLU, ELU, GroupNorm

"""
Martin Leipert
martin.leipert@th-deg.de

Backbone for Mask-RCNN with Residual SE Net
=> Most suessful in Shoe Segmentation
"""


def conv3d(in_channels, out_channels, kernel_size, bias, padding):
    return Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', LeakyReLU(inplace=True)))
        elif char == 'e':
            modules.append(('ELU', ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv', conv3d(in_channels, out_channels, kernel_size, bias, padding=padding)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding):
            self.add_module(name, module)


class ExcitationModule(Module):

    def __init__(self, channels, squeeze=2,  **kwargs):
        """
        Instanciate an excitation Block for channel Weighting
        :param channels: In Channel
        :param squeeze: Squeeze factor, determines how the input channels are reduced
        :param kwargs:
        """
        super(ExcitationModule, self).__init__(**kwargs)
        self.channels = channels
        self.squeeze_channels = int(channels / squeeze)

        # Excitation path
        self.add_module("pooling", AdaptiveAvgPool3d((1, 1, 1)))
        self.add_module("fc_1", Linear(self.channels, self.squeeze_channels))
        self.add_module("ex_relu", ReLU())
        self.add_module("fc_2", Linear(self.squeeze_channels, self.channels))
        self.add_module("ex_sigmoid", Sigmoid())

    def forward(self, x):
        # Scale part after residual
        scale = self.pooling(x)
        scale = scale.view(-1, self.channels)
        scale = self.fc_1(scale)
        scale = self.ex_relu(scale)
        scale = self.fc_2(scale)
        scale = self.ex_sigmoid(scale)

        return scale


class ResSEBlock(Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, order='gcr', num_groups=8, **kwargs):
        super(ResSEBlock, self).__init__()

        if in_channels % num_groups == 0:
            self.add_module("conv1", SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order,
                                                num_groups=num_groups))
        else:
            self.add_module("conv1", SingleConv(in_channels, out_channels, kernel_size=kernel_size, order=order))

        self.add_module("conv2", SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order,
                                            num_groups=num_groups))

        self.add_module("conv3", SingleConv(out_channels, out_channels, kernel_size=kernel_size, order=order,
                                            num_groups=num_groups))
        self.add_module("non_linearity", ReLU())

        # Exictation module for scaling
        self.add_module("excitation", ExcitationModule(out_channels))

    def forward(self, x):

        x = self.conv1(x)
        residual = x

        # residual block
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + residual
        x = self.non_linearity(x)

        # Scale part after residual
        scale = self.excitation(x)
        x = mul(x, scale.view((scale.shape[0], scale.shape[1], 1, 1, 1)))

        return x
