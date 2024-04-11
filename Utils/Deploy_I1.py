import cv2
import torch
from torch.utils.data.dataset import Dataset
# from Deep_Learning.Models.Segformer_UNet.model import SegFormer
from Deep_Learning.Models.Segformer_UNet_Concise.model import SegFormerUNetConcise
# from Deep_Learning.Models.Segformer_OutConv.model import SegFormer
from Deep_Learning.Data_Readers.Deploy_Reader_I1 import DataLoader
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def pth_push(image_path, model_path):
    file_path = "test logs.txt"
    file = open(file_path, "w", encoding="utf-8")
    image_path_list = []

    model= SegFormerUNetConcise(num_classes=1, phi="b3",in_channel=5)

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
    for image in data:
        image = image.to(device=device, dtype=torch.float32)

        pred = model(image)
        # pred = model(image)

        pred = torch.sigmoid(pred)

        pred[pred  > 0.1] = 1
        pred[pred <= 0.1] = 0

        pred = np.array(pred.data.cpu())
        cv2.imshow("pred", pred[0][0])
        cv2.waitKey(0)
        step += 1

if __name__ == "__main__":
    image_path = r"D:\Github_Repo\validate_ori_image\used"
    model_path = (r"D:\Github_Repo\Deep_Learning\Training_Strategies\model.pth")
    pth_push(image_path, model_path)