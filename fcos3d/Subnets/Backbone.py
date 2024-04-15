from fcos3d.Subnets.Blocks.ResSEBlock import ResSEBlock
from torch.nn import Module, ModuleList, Conv3d, ReLU, BatchNorm3d, MaxPool3d

"""
Martin Leipert
martin.leipert@th-deg.de
02.09.2022

Implementation of the Residual SE Backbone
"""


class ResSELayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, activation=ReLU,
                 down_conv=True, **kwargs):
        super(ResSELayer, self).__init__()

        self.down_conv = down_conv

        self.add_module("res_se_block",
                        ResSEBlock(in_channels, out_channels, kernel_size=kernel_size, order=order,
                                   num_groups=num_groups, activation=activation))

        if down_conv is True:
            self.add_module("stride_conv",
                            Conv3d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
                                   stride=(2, 2, 2), padding=(1, 1, 1)))
        else:
            self.add_module("max_pool", MaxPool3d(3, 2, 1))

    def forward(self, x):

        x = self.res_se_block(x)

        if self.down_conv is True:
            x = self.stride_conv(x)
        else:
            x = self.max_pool(x)

        return x


class ResSEInLayer(Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, activation=ReLU,
                     down_conv=True, **kwargs):
        super(ResSEInLayer, self).__init__()

        self.down_conv = down_conv

        self.add_module("batch_norm", BatchNorm3d(in_channels))

        self.add_module("conv", Conv3d(in_channels, num_groups, kernel_size=(3, 3, 3), padding=(1, 1, 1)))

        self.add_module("res_se_block",
                        ResSEBlock(num_groups, out_channels, kernel_size=kernel_size, order=order,
                                   num_groups=num_groups, activation=activation))

        if down_conv is True:
            self.add_module("stride_conv",
                            Conv3d(out_channels, out_channels, kernel_size=(kernel_size, kernel_size, kernel_size),
                                   stride=(2, 2, 2), padding=(1, 1, 1)))
        else:
            self.add_module("max_pool", MaxPool3d(3, 2, 1))

    def forward(self, x):

        x = self.batch_norm(x)
        x = self.conv(x)
        x = self.res_se_block(x)

        if self.down_conv is True:
            x = self.stride_conv(x)
        else:
            x = self.max_pool(x)

        return x


class ResSEBackbone(Module):

    def __init__(self, f_maps, in_channels, levels=5, out_levels=3):
        super(ResSEBackbone, self).__init__()

        self.__levels = levels
        self.__out_levels = levels - out_levels
        self._out_channels = []

        layers = ModuleList()

        for level in range(levels):

            f_maps_in = in_channels if level == 0 else f_maps * pow(2, level)
            f_maps_out = f_maps * pow(2, level + 1)

            if level > 0:
                layers.append(ResSELayer(f_maps_in, f_maps_out, down_conv=True))
            else:
                layers.append(ResSEInLayer(f_maps_in, f_maps_out, down_conv=True))

            self._out_channels.append(f_maps_out)

        self.add_module("layer_list", layers)

    def forward(self, x):

        x_out = []

        for index, layer in enumerate(self.layer_list):
            x = layer(x)

            if index >= self.__out_levels:
                x_out.append(x)

        return x, x_out

    def get_out_layers(self):
        return self._out_channels[self.__out_levels:]

