import glob
import os

import numpy as np
import torch
from torch.utils.data.dataset import Dataset
from torch.utils.tensorboard import SummaryWriter

from Data_Readers import Data_Reader
import cv2

class DataLoader(Dataset):
    def __init__(self,label_path_list, reverse_channel = True):
        super(DataLoader, self).__init__()
        self.label_path  = label_path_list
        self.reverse_channel = reverse_channel

    def __getitem__(self, index):

        #根据index读取图像和标签
        label_path = self.label_path[index]
        image1_path = label_path.replace("label", "image1")
        image2_path = label_path.replace("label", "image2")

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
        image1_array = image1_array.reshape(image1_bands,image1_height,image1_width)
        if self.reverse_channel or image1_bands < 3:
            image1_array = image1_array[::-1,]
        image2_array = image2_array.reshape(image2_bands,image2_height,image2_width)
        label_array  = label_array .reshape(label_bands  ,label_height ,label_width)

        image1_array = cv2.normalize(image1_array,None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image2_array = cv2.normalize(image2_array, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        image1_array = cv2.normalize(image1_array, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)


        # 转为tensor
        image1_array = torch.tensor(image1_array,dtype=torch.float32)
        image2_array = torch.tensor(image2_array,dtype=torch.float32)
        label_array  = torch.tensor(label_array ,dtype=torch.float32)

        return image1_array, image2_array, label_array

    def __len__(self):
        return len(self.label_path)

if __name__ == "__main__":

    from Data_Distributors.Processed_Distributor import Distributor
    datasets = Distributor(r"D:\Project\CUMT_PAPER_DATASETS_FINAL").N_Cross_Distributor()
    for i in range(3):
        validate_dataloader = DataLoader(datasets[i][1])
        train_dataloader = DataLoader(datasets[i][0])
        print(len(train_dataloader))
        print(len(validate_dataloader))


