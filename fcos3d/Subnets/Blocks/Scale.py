from torch.nn import Module, Parameter
import torch


class Scale(Module):
    def __init__(self, init=1.0):
        super().__init__()

        self.scale = Parameter(torch.tensor([init], dtype=torch.float32))

    def forward(self, input):
        return input * self.scale