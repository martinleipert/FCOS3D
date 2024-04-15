import torch
from fcos3d.FCOS3D import FCOS3D
from fcos3d.Utils.Structures.BoxList import BoxList


def main():

    # Let it run with an empty tensor
    fcos = FCOS3D(device="cpu")
    fcos.eval()

    in_tensor = torch.ones([1, 1, 128, 128, 128])

    # Forward test
    result = fcos(in_tensor)

    # Now use a tensor with item

    in_tensor = torch.zeros([1, 1, 128, 128, 128])
    in_tensor[0, 0, 32:96, 32:96, 32:96] = torch.ones([64, 64, 64])

    targets = BoxList(torch.tensor([[[32, 32, 32, 96, 96, 96]]]), torch.tensor([128, 128, 128]))
    targets.add_field("labels", torch.tensor([[1]]))
    targets.add_field("centers", torch.tensor([[[64, 64, 64]]]))
    targets.add_field("masks", torch.clone(in_tensor))

    fcos.train()
    result = fcos(in_tensor, targets)

    loss = result[0] + result[1] + result[2]
    loss.backward()

    pass


if __name__ == "__main__":
    main()
