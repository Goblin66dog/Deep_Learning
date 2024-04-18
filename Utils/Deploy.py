import glob

import cv2
import torch
from Data_Readers.Data_Reader import Dataset
import os
import numpy as np
from osgeo import gdal
from Data_Processors.Padding import Padding
from Data_Processors.Percent_Linear_Enhancement import PercentLinearEnhancement
from Data_Processors.Flip8x import Flip8x

from Models.UNet.model import                     UNet
from Models.AG_UNet.model import                  AGUNet
from Models.ASPP_U2Net.model import               ASPPU2Net
from Models.DeepLab_V3_Plus.model import          DeepLab
from Models.SegFormer.model import                SegFormer
from Models.SegFormer_OutConv.model import        SegFormerOutConv
from Models.SegFormer_UNet.model import           SegFormerUNet
from Models.SegFormer_UNet_Concise.model import   SegFormerUNetConcise
from Data_Readers import Deploy_Loader_I1, Deploy_Loader_I2


import warnings
warnings.filterwarnings("ignore")


class DeployByPth:
    def __init__(self, model, model_path, output_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(device=self.device)
        self.model = model.eval()
        self.output_path = output_path

    @staticmethod
    def extract_name(file_path):
        name = os.path.splitext(os.path.basename(file_path))[0]
        return name

    def SaveWithGeoInfo(self, item, image, image_path):
        axs = [0,1],[0,1]

        image = np.transpose(image, axs[0])
        driver = gdal.GetDriverByName('GTiff')
        dataset = driver.Create(self.output_path + "\\" + self.extract_name(file_path=image_path)+ "prediction.TIF",
                                image.shape[1],
                                image.shape[0],
                                1,
                                gdal.GDT_Float32)
        image = np.transpose(image, axs[1])
        dataset.SetGeoTransform(item.geotrans)  # 写入仿射变换参数
        dataset.SetProjection(item.proj)  # 写入投影
        dataset.GetRasterBand(1).WriteArray(image)
        dataset.FlushCache()  # 确保所有写入操作都已完成
        dataset = None

    def I1(self, input1_path, image_shape="CHW", mode="batch"):
        if mode != "batch":
            #Dataset类
            item1 = Dataset(input1_path)
            #栅格图像
            input1 = item1.array
            #预处理
            input1 = PercentLinearEnhancement(input1, image_shape).gray_process()
            input1 = Padding(input1, image_shape).min(8)
            input1 = cv2.normalize(input1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            input1 = torch.tensor(input1, dtype=torch.float32)
            # 模型预测
            input1 = input1.to(device=self.device, dtype=torch.float32)
            output = self.model(input1)
            #激活函数
            output = torch.sigmoid(output)
            output[output > .5] = 1
            output[output <= .5] = 0
            #图像转到cpu
            output = np.array(output.data.cpu())
            #保存
            self.SaveWithGeoInfo(item1, output, input1_path)

        else:
            #获取文件路径列表
            input1_path_list = glob.glob(os.path.join(input1_path, "image/*"))
            #循环遍历
            for path in input1_path_list:
                # Dataset类
                item1 = Dataset(path)
                # 栅格图像
                input1 = item1.array
                # 预处理
                input1 = PercentLinearEnhancement(input1, image_shape).gray_process()
                input1 = Padding(input1, image_shape).min(8)
                input1 = cv2.normalize(input1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                input1 = torch.tensor(input1, dtype=torch.float32)
                # 模型预测
                input1 = input1.to(device=self.device, dtype=torch.float32)
                output = self.model(input1)
                # 激活函数
                output = torch.sigmoid(output)
                output[output > .5] = 1
                output[output <= .5] = 0
                # 图像转到cpu
                output = np.array(output.data.cpu())
                # 保存
                self.SaveWithGeoInfo(item1, output, input1_path)

    def I2_single(self, input1_path, input2_path, input1_shape="CHW", input2_shape="CHW"):
        # Dataset类
        item1 = Dataset(input1_path)
        item2 = Dataset(input2_path)
        # 栅格图像
        input1 = item1.array
        input2 = item2.array
        # 预处理
        input1 = input1.reshape(item1.bands, item1.height, item1.width)
        input1 = input1[::-1, ]
        input2 = input2.reshape(item2.bands, item2.height, item2.width)
        input1 = PercentLinearEnhancement(input1, input1_shape).gray_process()
        input2 = PercentLinearEnhancement(input2, input2_shape).gray_process()
        input1 = Padding(input1, input1_shape).min(8)
        input2 = Padding(input2, input2_shape).min(8)
        input1 = cv2.normalize(input1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        input2 = cv2.normalize(input2, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        input1 = torch.tensor(input1, dtype=torch.float32)
        input2 = torch.tensor(input2, dtype=torch.float32)
        input1 = input1.reshape(1, 4, input1.shape[1], input1.shape[2])
        input2 = input2.reshape(1, 1, input2.shape[1], input2.shape[2])
        # 模型预测
        input1 = input1.to(device=self.device, dtype=torch.float32)
        input2 = input2.to(device=self.device, dtype=torch.float32)

        output = self.model(
            torch.cat([input1, input2], dim=1).
            to(device=self.device, dtype=torch.float32)
            ,input2
        )
        # 激活函数
        output = torch.sigmoid(output)
        output[output > .5] = 1
        output[output <= .5] = 0
        # 图像转到cpu
        output = np.array(output.data.cpu())[0][0]
        # 保存
        self.SaveWithGeoInfo(item1, output, input1_path)

    def I2_batch(self, input_path, input1_shape="CHW", input2_shape="CHW"):
        # 获取文件路径列表
        input1_path_list = glob.glob(os.path.join(input_path, "image1/*"))
        input2_path_list = glob.glob(os.path.join(input_path, "image2/*"))
        if len(input1_path_list) == 0 or len(input2_path_list) == 0:
            print("没读进来哟~~")
        # 循环遍历
        for path in range(len(input1_path_list)):
            self.I2_single(input1_path_list[path],
                           input2_path_list[path])

if __name__ == "__main__":
    # model1 = UNet(in_channels=5,num_classes=1)
    # model2 = AGUNet(in_channels=5,num_classes=1)
    # model3 = ASPPU2Net(image_channels=4,texture_channels=1,num_classes=1)
    # model4 = DeepLab(in_channels=5, num_classes=1)
    # model5 = SegFormer(in_channels=5,num_classes=1,backbone="b3")
    # model6 = SegFormerOutConv(in_channels=5,num_classes=1,backbone="b3")
    # model7 = SegFormerUNet(in_channels=5,num_classes=1,backbone="b3")
    model8 = SegFormerUNetConcise(in_channels=5,num_classes=1,backbone="b3")
    model_path = r"D:\Github_Repo\Open Pit Mining\logs\SegFormer_U\model logs\model2.pth"
    output_path= r"D:\Github_Repo\Open Pit Mining\尝试"
#     DeployByPth(model8, model_path,output_path).I2_single(
#         r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.13\image1\2020.tif",
#         r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.13\image2\2020.tif"
# )
    DeployByPth(model8, model_path, output_path).I2_batch(
        r"D:\Github_Repo\Open Pit Mining\validate_ori_image",
    )
