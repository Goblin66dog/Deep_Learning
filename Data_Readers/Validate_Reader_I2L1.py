import glob
import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from Deep_Learning.Data_Processors.Padding import Padding
from Deep_Learning.Data_Processors.Percent_Linear_Enhancement import PercentLinearEnhancement

from Deep_Learning.Data_Readers import Data_Reader
import cv2

class DataLoader(Dataset):
    def __init__(self, input_datasets_path):
        super(DataLoader, self).__init__()
        self.image1_path = glob.glob(os.path.join(input_datasets_path, "image1/*"))
        self.image2_path = glob.glob(os.path.join(input_datasets_path, "image2/*"))
        self.label_path = glob.glob(os.path.join(input_datasets_path, "label/*"))

    def __getitem__(self, index):

        #根据index读取图像和标签
        image1_path = self.image1_path[index]
        image2_path = self.image2_path[index]
        label_path = self.label_path[index]

        #读取训练图片和标签图片
        image1 = Data_Reader.Dataset(image1_path)
        image2 = Data_Reader.Dataset(image2_path)
        label = Data_Reader.Dataset(label_path)

        image1_array = image1.array
        image1_width = image1.width
        image1_height= image1.height
        image1_bands = image1.bands
        del image1
        image2_array = image2.array
        image2_width = image2.width
        image2_height= image2.height
        image2_bands = image2.bands
        del image2
        label_array = label.array
        label_width = label.width
        label_height= label.height
        label_bands = label.bands
        del label

        #reshape
        image1_array = image1_array.reshape(image1_bands, image1_height, image1_width)
        image1_array = image1_array[::-1, ]
        image2_array = image2_array.reshape(image2_bands, image2_height, image2_width)
        label_array = label_array.reshape(label_bands, label_height, label_width)

        # image1_array = PercentLinearEnhancement(image1_array, image_shape="CHW").gray_process()
        # image2_array = PercentLinearEnhancement(image2_array, image_shape="HW").gray_process()
        # image1_array = Padding(image1_array, image_shape="CHW").nor(256, 256)
        # image2_array = Padding(image2_array, image_shape="CHW").nor(256, 256)
        # label_array = Padding(label_array, image_shape="CHW").nor(256, 256)


        image1_array = cv2.normalize(image1_array,None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image2_array = cv2.normalize(image2_array, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        label_array  = cv2.normalize(label_array, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        pixels = np.sum([label_array == 1])

        # 转为tensor
        image1_array = torch.tensor(image1_array,dtype=torch.float32)
        image2_array = torch.tensor(image2_array,dtype=torch.float32)
        label_array  = torch.tensor(label_array ,dtype=torch.float32)

        return image1_array, image2_array, label_array, label_path, pixels

    def __len__(self):
        return len(self.image1_path)

if __name__ == "__main__":

    train_dataloader = DataLoader(r"D:\Project\CUMT_PAPER_DATASETS_FINAL")
    train_data = torch.utils.data.DataLoader(
        dataset=train_dataloader,
        batch_size=1,
        shuffle=True
    )
    # # writer = {
    # #     "data":SummaryWriter(r"logs")
    # # }
    step = 1
    for x, y, z in train_data:
        print(x.shape, step)
        # writer["data"].add_image1s("image1", image1, step)
        # writer["data"].add_image1s("label", label, step)

        step += 1
    # # writer["data"].close()

