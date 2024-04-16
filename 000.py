import torch

from Models.DeepLab_V3_Plus.model import DeepLab
from Models.SegFormer_OutConv.model import SegFormer
# from Models.SegFormer.model import SegFormer


input = torch.ones([4,4,512,512]).to("cuda")
model = SegFormer(1,"b1", False,in_channels=4)
model.to("cuda")

output = model(input)

print(output.shape)