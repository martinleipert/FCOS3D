
from torch.nn import Module, ModuleList, Conv3d, ReLU, BatchNorm3d, GroupNorm, Sequential
from torch.nn.functional import interpolate
from fcos3d.Utils.Utility import init_conv_kaiming, init_conv_std
from torch import cat


class FPNModule(Module):

    def __init__(self, features_in, f_maps_inout, num_groups=8):
        super(FPNModule, self).__init__()

        self.features_in = features_in
        self.f_maps_inout = f_maps_inout

        self.add_module("conv1a", Conv3d(features_in, f_maps_inout, kernel_size=(3, 3, 3), groups=num_groups,
                                         padding=(1, 1, 1), dilation=(1, 1, 1)))
        self.add_module("conv1b", Conv3d(features_in, f_maps_inout, kernel_size=(3, 3, 3), groups=num_groups,
                                         padding=(2, 2, 2), dilation=(2, 2, 2)))
        self.add_module("conv2", Conv3d(f_maps_inout * 2, f_maps_inout * 2, kernel_size=(1, 1, 1), groups=num_groups))

    def forward(self, x):
        x_a = self.conv1a(x)
        x_b = self.conv1b(x)
        x = cat((x_a, x_b), dim=1)
        x = self.conv2(x)

        return x


class ResSEFPN(Module):

    def __init__(self, f_maps_inout, levels=3, add_levels=2, num_groups=8):
        super(ResSEFPN, self).__init__()

        self.__levels = levels
        self.__add_levels = add_levels

        self.p_layers = ModuleList()
        self.l_layers = ModuleList()

        for index in range(levels, 0, -1):
            p_name = f"p{index + add_levels}"

            # layer_list.add_module(name, FPNModule(features_in, f_maps_inout, num_groups=num_groups))
            self.p_layers.add_module(p_name, Sequential(
                Conv3d(f_maps_inout, f_maps_inout, kernel_size=(3, 3, 3), groups=num_groups, padding=(1, 1, 1)),
            ))

            # ResSEBlock(f_maps_inout, f_maps_inout)))

        for index in range(levels):
            features_in = int(pow(2, index) * f_maps_inout)
            l_name = f"l{index + add_levels}"
            self.l_layers.add_module(l_name, Sequential(
                Conv3d(features_in, f_maps_inout, kernel_size=(1, 1, 1), groups=num_groups),
            ))

        self.add_layers = ModuleList()

        for index in range(add_levels):
            name = f"p{index + (levels + add_levels)}"

            features_in = f_maps_inout * pow(2, levels + index - 1)  # pow(2, levels+index-1) * f_maps_inout

            if index == 0:
                self.add_layers.add_module(name, Sequential(
                    Conv3d(features_in, f_maps_inout, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                           groups=num_groups)))
            else:
                self.add_layers.add_module(name, Sequential(
                    ReLU(),
                    Conv3d(f_maps_inout, f_maps_inout, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1),
                           groups=num_groups)))
            # ,
            # ResSEBlock(f_maps_inout, f_maps_inout)))

        self.apply(init_conv_kaiming)

    def forward(self, x, x_bb):

        x_out = []
        x = x_bb[-1]
        for index, layer in enumerate(self.add_layers):
            x = layer(x)
            x_out.append(x)

        x_out = list(reversed(x_out))

        x_lateral = []
        for index, layer in enumerate(self.l_layers):

            if index < self.__levels:
                x_in = x_bb[index]
                x_l = layer(x_in)
                x_lateral.append(x_l)
            # else:
            #     x_in = x_out[index-self.__levels]
            #     x_l = layer(x_in)
            #     x_lateral.append(x_l)

        for index, layer in enumerate(self.p_layers):
            x_in = x_lateral[self.__levels - index - 1]

            if index == 0:
                x = layer(x_in)
            else:
                x = interpolate(x, scale_factor=(2, 2, 2), mode="nearest")
                x = x_in + x
                x = layer(x)
            x_out.append(x)

        return list(reversed(x_out))
