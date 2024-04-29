import torch
from sklearn.metrics import recall_score, precision_score,accuracy_score
from torch.utils.data.dataset import Dataset

from Models.UNet.model import                     UNet
from Models.AG_UNet.model import                  AGUNet
from Models.ASPP_U2Net.model import               ASPPU2Net
from Models.DeepLab_V3_Plus.model import          DeepLab
from Models.SegFormer.model import                SegFormer
from Models.SegFormer_OutConv.model import        SegFormerOutConv
from Models.SegFormer_UNet.model import           SegFormerUNet
from Models.SegFormer_UNet_Concise.model import   SegFormerUNetConcise

from Data_Loaders.Validate_Reader_I2L1 import DataLoader
import numpy as np

import warnings
warnings.filterwarnings("ignore")

def pth_push(image_path, model_path):
    file_path = "test logs.txt"
    file = open(file_path, "w", encoding="utf-8")
    image_path_list = []

    model1 = UNet(in_channels=5, num_classes=1)
    model2 = AGUNet(in_channels=5, num_classes=1)
    model3 = ASPPU2Net(image_channels=4, texture_channels=1, num_classes=1)
    model4 = DeepLab(in_channels=5, num_classes=1)
    model5 = SegFormer(in_channels=5, num_classes=1, backbone="b3")
    model6 = SegFormerOutConv(in_channels=5, num_classes=1, backbone="b3")
    model7 = SegFormerUNet(in_channels=5, num_classes=1, backbone="b3")
    model8 = SegFormerUNetConcise(in_channels=5, num_classes=1, backbone="b3")
    model = model1

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
    file.writelines("      " +"recall" + "   " + "precision" + "\n")

    recall_weight = []
    precision_weight = []
    recall_list = []
    precision_list = []
    accuracy = []


    step = 0
    for image, texture, label, path, label_pixels in data:
        # if label_pixels == 0:
        #     continue
        image = torch.cat([image, texture], dim=1).to(device=device, dtype=torch.float32)
        # image = image.to(device=device, dtype=torch.float32)

        texture = texture.to(device=device, dtype=torch.float32)
        label = label.to(device=device, dtype=torch.float32)

        pred = model(image)
        # pred = model(image, texture)

        pred = torch.sigmoid(pred)

        pred[pred  > 0.5] = 1
        pred[pred <= 0.5] = 0

        pred = np.array(pred.data.cpu())
        label = np.array(label.data.cpu())
        # cv2.imshow("pred", pred[0][0])
        # cv2.waitKey(0)
        # cv2.imwrite("image"+str(step+1)+".tif",pred[0][0])
        pred_pixels = np.sum(pred>0)
        # if pred_pixels == 0:
        #     continue
        pred  = pred.reshape(-1)
        label = label.reshape(-1)

        recall = recall_score(label, pred)
        precision = precision_score(label, pred)
        acc       = accuracy_score(label,pred)

        step += 1
        file.writelines(format(str(step), '>03') +":"+format(str(round(recall, 4)), '<06') + "   "
                                      +format(str(round(precision, 4)), '<06') + "\n")

        recall_weight.append(label_pixels)
        precision_weight.append(pred_pixels)
        recall_list.append(recall)
        precision_list.append(precision)
        accuracy.append(acc)

        image_path_list.append(path)
    file.writelines("\n"+"各验证样本路径："+ "\n")

    step = 0
    for path in image_path_list:
        step += 1
        file.writelines(format(str(step), '>03') +": "+path[0] + "\n")

    file.writelines("整体精度："+"\n")
    recall_all_weight = sum(recall_weight)
    precision_all_weight = sum(precision_weight)

    recall = []
    precision = []
    for each_sample in range(len(recall_list)):
        recall.append(recall_list[each_sample]*(recall_weight[each_sample]/recall_all_weight))
        precision.append(precision_list[each_sample]*(precision_weight[each_sample]/precision_all_weight))

    precision = sum(precision)
    recall = sum(recall)
    f1     = 2*precision*recall/(precision+recall)
    accuracy = sum(accuracy)/len(accuracy)

    file.writelines("recall："+str(np.array(recall))+"\n")
    file.writelines("precision：" + str(precision) + "\n")
    file.writelines("F1:"+str(f1)+"\n")
    file.writelines("acc:" + str(accuracy) + "\n")
    file.close()
if __name__ == "__main__":
    image_path = r"D:\Github_Repo\Open Pit Mining\Deploy"
    # model_path = r"D:\Github_Repo\Open Pit Mining\logs\SegFormer_U\model logs\model2.pth"
    # model_path = r"D:\Github_Repo\Deep_Learning\Training_Strategies\model.pth"
    # model_path = r"D:\Github_Repo\Open Pit Mining\logs\SegFormer\model.pth"
    model_path = r"D:\Github_Repo\Open Pit Mining\logs\UNet\model.pth"
    # model_path = r"D:\Github_Repo\Open Pit Mining\logs\ASPPU2Net\model2.pth"
    pth_push(image_path, model_path)
