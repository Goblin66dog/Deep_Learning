import cv2
import torch
from torch.utils.data.dataset import Dataset

from Deep_Learning.Models.ASPP_U2Net.model import ASPPU2Net
# from Deep_Learning.Models.Segformer_UNet.model import SegFormer
from Deep_Learning.Models.Segformer_UNet_Simplifier.model import SegFormer
# from Deep_Learning.Models.Segformer.model import SegFormer
from Deep_Learning.Data_Readers.I2_DEPLOY import DataLoader
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def pth_push(image_path, model_path):
    file_path = "test logs.txt"
    file = open(file_path, "w", encoding="utf-8")
    image_path_list = []

    model= SegFormer(num_classes=1, phi="b3",in_channel=5)
    # model = UNet(classes=1,channels=5)
    # model = AGUNet(classes=1, channels=5)
    model = ASPPU2Net(image_channels=4,texture_channels=1,classes=1)

    model_path = model_path#pth权重文件地址
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')#cpu or gpu
    model.load_state_dict(torch.load(model_path, map_location=device))#加载pth文件
    model.to(device=device)
    model  = model.eval()

    dataloader = DataLoader(image_path)
    data = torch.utils.data.DataLoader(
        dataset=dataloader,
        batch_size=1,
        shuffle=False
    )


    step = 0
    for image, texture in data:
        image = torch.cat([image, texture], dim=1).to(device=device, dtype=torch.float32)
        # image = image.to(device=device, dtype=torch.float32)
        texture = texture.to(device=device, dtype=torch.float32)

        pred = model(image, texture)
        # pred = model(image)

        pred = torch.sigmoid(pred)

        pred[pred  > 0.5] = 1
        pred[pred <= 0.5] = 0

        pred = np.array(pred.data.cpu())
        cv2.imshow("pred", pred[0][0])
        cv2.waitKey(0)
        step += 1

if __name__ == "__main__":
    image_path = r"D:\Github_Repo\Deploy"
    model_path = (r"D:\Github_Repo\logs\SegFormer_U\model logs\model.pth")
    pth_push(image_path, model_path)
