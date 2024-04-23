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
import matplotlib.pyplot as plt


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

# class HeatMap:
#     def __init__(self, model, model_path):
#         self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#         model.load_state_dict(torch.load(model_path, map_location=self.device))
#         model.to(device=self.device)
#         self.model = model.eval()
#
#
#     def I1(self, input1_path, image_shape="CHW", mode="batch"):
#         if mode != "batch":
#             #Dataset类
#             item1 = Dataset(input1_path)
#             #栅格图像
#             input1 = item1.array
#             #预处理
#             input1 = PercentLinearEnhancement(input1, image_shape).gray_process()
#             input1 = Padding(input1, image_shape).min(8)
#             input1 = cv2.normalize(input1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#             input1 = torch.tensor(input1, dtype=torch.float32)
#             # 模型预测
#             input1 = input1.to(device=self.device, dtype=torch.float32)
#             output = self.model(input1)
#             #激活函数
#             output = torch.sigmoid(output)
#             # output[output > .5] = 1
#             # output[output <= .5] = 0
#             #图像转到cpu
#             output = np.array(output.data.cpu())
#             #保存
#             plt.imshow(output, cmap="jet")
#             plt.axis('off')
#             plt.xticks([])
#             plt.yticks([])
#             plt.show()
#
#     def I2(self, input1_path, input2_path, input1_shape="CHW", input2_shape="CHW"):
#         # Dataset类
#         item1 = Dataset(input1_path)
#         item2 = Dataset(input2_path)
#         # 栅格图像
#         input1 = item1.array
#         input2 = item2.array
#         # 预处理
#         input1 = input1.reshape(item1.bands, item1.height, item1.width)
#         input1 = input1[::-1, ]
#         input2 = input2.reshape(item2.bands, item2.height, item2.width)
#         input1 = PercentLinearEnhancement(input1, input1_shape).gray_process()
#         input2 = PercentLinearEnhancement(input2, input2_shape).gray_process()
#         input1 = Padding(input1, input1_shape).min(8)
#         input2 = Padding(input2, input2_shape).min(8)
#         input1 = cv2.normalize(input1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         input2 = cv2.normalize(input2, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
#         input1 = torch.tensor(input1, dtype=torch.float32)
#         input2 = torch.tensor(input2, dtype=torch.float32)
#         input1 = input1.reshape(1, 4, input1.shape[1], input1.shape[2])
#         input2 = input2.reshape(1, 1, input2.shape[1], input2.shape[2])
#         # 模型预测
#         input1 = input1.to(device=self.device, dtype=torch.float32)
#         input2 = input2.to(device=self.device, dtype=torch.float32)
#
#         output = self.model(
#             torch.cat([input1, input2], dim=1).
#             to(device=self.device, dtype=torch.float32)
#             # input1
#             # ,input2
#         )
#         # 激活函数
#         output = torch.sigmoid(output)
#         # output[output > .5] = 1
#         # output[output <= .5] = 0
#         # 图像转到cpu
#         output = np.array(output.data.cpu())[0][0]
#         # 保存
#         plt.imshow(output, cmap="jet")
#         plt.axis('off')
#         plt.xticks([])
#         plt.yticks([])
#         plt.show()

def I2(model, input1_path, input2_path, model_num, input1_shape="CHW", input2_shape="CHW"):
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
    input1 = Padding(input1, input1_shape).min(16)
    input2 = Padding(input2, input2_shape).min(16)
    input1 = cv2.normalize(input1, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input2 = cv2.normalize(input2, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    input1 = torch.tensor(input1, dtype=torch.float32)
    input2 = torch.tensor(input2, dtype=torch.float32)
    input1 = input1.reshape(1, 4, input1.shape[1], input1.shape[2])
    input2 = input2.reshape(1, 1, input2.shape[1], input2.shape[2])
    # 模型预测
    input1 = input1.to(device="cuda", dtype=torch.float32)
    input2 = input2.to(device="cuda", dtype=torch.float32)
    if model_num==0:
        output = model(
            torch.cat([input1, input2], dim=1).
            to(device="cuda", dtype=torch.float32)
            # input1
            # ,input2
        )
    elif model_num == 1:
        output = model(
            # torch.cat([input1, input2], dim=1).
            # to(device="cuda", dtype=torch.float32)
            input1
            ,input2
        )
    elif model_num == 2:
        output = model(
            torch.cat([input1, input2], dim=1).
            to(device="cuda", dtype=torch.float32)
            # input1
            # ,input2
        )
    else:
        output = model(
            torch.cat([input1, input2], dim=1).
            to(device="cuda", dtype=torch.float32)
            # input1
            , input2
        )
    # 激活函数
    output = torch.sigmoid(output)
    # output[output > .5] = 1
    # output[output <= .5] = 0
    # 图像转到cpu
    output = np.array(output.data.cpu())[0][0]
    # 保存
    return output
    # plt.imshow(output, cmap="jet")
    # plt.axis('off')
    # plt.xticks([])
    # plt.yticks([])
    # plt.show()



if __name__ == "__main__":
    model1 = UNet(in_channels=5,num_classes=1)
    # model2 = AGUNet(in_channels=5,num_classes=1)
    model3 = ASPPU2Net(image_channels=4,texture_channels=1,num_classes=1)
    # model4 = DeepLab(in_channels=5, num_classes=1)
    # model5 = SegFormer(in_channels=5,num_classes=1,backbone="b3")
    model6 = SegFormerOutConv(in_channels=5,num_classes=1,backbone="b3")
    # model7 = SegFormerUNet(in_channels=5,num_classes=1,backbone="b3")
    model8 = SegFormerUNetConcise(in_channels=5,num_classes=1,backbone="b3")
    model_path1 = r"D:\Github_Repo\Open Pit Mining\logs\UNet\model.pth"
    model_path3 = r"D:\Github_Repo\Open Pit Mining\logs\ASPPU2Net\model2.pth"
    model_path6 = r"D:\Github_Repo\Open Pit Mining\logs\SegFormer\model.pth"
    model_path8 = r"D:\Github_Repo\Open Pit Mining\logs\SegFormer_U\model logs\model2.pth"
    # HeatMap(model1, model_path).I2(
    #             r"D:\Github_Repo\Open Pit Mining\validate_ori_image\image1\99.TIF",
    #             r"D:\Github_Repo\Open Pit Mining\validate_ori_image\image2\99.TIF"
    # )
    model_path_list = [model_path1, model_path3, model_path6, model_path8]
    model_list = [model1, model3, model6, model8]

    plt.figure(dpi=400)
    plt.subplots_adjust(wspace=-0.81, hspace=0.02)
    num = 1
    rows = 6

    image = Dataset(r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\image1\30.TIF").array
    image = Padding(image, "CHW").min(16)
    image = image[::-1,]
    image = np.transpose(image[1:], [1,2,0])
    # image = PercentLinearEnhancement(image, "CHW").gray_process()
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.subplot(rows, 3, num)
    plt.imshow(image)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    num += 1
    image = Dataset(r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\image1\05.TIF").array
    image = Padding(image, "CHW").min(16)
    image = image[::-1, ]
    image = np.transpose(image[1:], [1,2,0])
    # image = PercentLinearEnhancement(image, "CHW").gray_process()
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.subplot(rows, 3, num)
    plt.imshow(image)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    num += 1
    image = Dataset(r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\image1\99.TIF").array
    image = Padding(image, "CHW").min(16)
    image = image[::-1, ]
    image = np.transpose(image[1:], [1,2,0])
    # image = PercentLinearEnhancement(image, "CHW").gray_process()
    image = cv2.normalize(image, None, 0, 1, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    plt.subplot(rows, 3, num)
    plt.imshow(image)
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    num += 1



    image = Dataset(r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\label\30.TIF").array
    image = Padding(image, "HW").min(16)
    plt.subplot(rows, 3, num)
    plt.imshow(image,"gray")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    num += 1
    image = Dataset(r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\label\05.tif").array
    image = Padding(image, "HW").min(16)
    plt.subplot(rows, 3, num)
    plt.imshow(image,"gray")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    num += 1
    image = Dataset(r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\label\99.TIF").array
    image = Padding(image, "HW").min(16)
    plt.subplot(rows, 3, num)
    plt.imshow(image,"gray")
    plt.axis('off')
    plt.xticks([])
    plt.yticks([])
    num += 1




    for i in range(len(model_path_list)):
        model = model_list[i]
        model_path = model_path_list[i]
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device=device)
        model = model.eval()
        print()
        ax0 = I2(model,
           r"D:\Github_Repo\Open Pit Mining\validate_ori_image\image1\30.TIF",
           r"D:\Github_Repo\Open Pit Mining\validate_ori_image\image2\30.TIF",
           model_num=i)
        ax1 = I2(model,
           r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\image1\05.tif",
           r"D:\Github_Repo\Open Pit Mining\validate_ori_image\4.12(1)\image2\05.tif",
           model_num=i)
        ax2 = I2(model,
           r"D:\Github_Repo\Open Pit Mining\validate_ori_image\image1\99.TIF",
           r"D:\Github_Repo\Open Pit Mining\validate_ori_image\image2\99.TIF",
           model_num=i)
        plt.subplot(rows,3,  num)
        plt.imshow(ax0, cmap="coolwarm")
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        num +=1
        plt.subplot(rows, 3, num)
        plt.imshow(ax1, cmap="coolwarm")
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        num +=1
        plt.subplot(rows, 3, num)
        plt.imshow(ax2, cmap="coolwarm")
        plt.axis('off')
        plt.xticks([])
        plt.yticks([])
        num +=1
    # plt.show()
    plt.savefig(r"C:\Users\Vitch\Desktop\Figure_2.png",dpi=500,bbox_inches='tight')
