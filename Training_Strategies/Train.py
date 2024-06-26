import warnings
import random

import numpy as np
import torch.nn as nn
import torch.utils.data
from sklearn.metrics import recall_score, precision_score
from torch import optim
from torch.utils.tensorboard import SummaryWriter


from Data_Loaders.I2L1 import DataLoader
from Models.UNet.model                    import UNet
from Models.DeepLab_V3_Plus.model         import DeepLab
from Models.AG_UNet.model                 import AGUNet
from Models.ASPP_U2Net.model              import ASPPU2Net
from Models.SegFormer.model               import SegFormer
from Models.SegFormer_OutConv.model       import SegFormerOutConv
from Models.SegFormer_UNet.model          import SegFormerUNet
from Models.SegFormer_UNet_Concise.model  import SegFormerUNetConcise
from Data_Distributors import Random_Distributor, Alloted_Distributor, N_Cross_Distributor, Processed_Distributor



warnings.filterwarnings("ignore")

# def train(device, epochs=10, batch_size=2, lr=5e-3,step_size_up=1):
#     # net = SegFormer(num_classes=1,pretrained=False,in_channel=5,phi="b3")
#     net = ASPPU2Net(image_channels=4, texture_channels=1,num_classes=1)
# ########################################################################################################################
#     # pretrained_dict = torch.load(r"D:\Github_Repo\logs\SegFormer\model.pth",
#     #                              map_location=device)
#     # model_dict = net.state_dict()
#     # a = {}
#     # for k, v in pretrained_dict.items():
#     #     try:
#     #         if np.shape(model_dict[k]) == np.shape(v):
#     #             a[k] = v
#     #     except:
#     #         pass
#     # model_dict.update(a)
#     # net.load_state_dict(model_dict)
#     # print('Finished!')
#     #
#     # for param in net.backbone.parameters():
#     #     param.requires_grad = False
#     # for param in net.decode_head.parameters():
#     #     param.requires_grad = False
# ########################################################################################################################
#     net.to(device=local_device)
#     writer = {
#         "loss": SummaryWriter(r"logs\loss"),
#         'recall': SummaryWriter(r"logs\recall"),
#         'precision': SummaryWriter(r"logs\precision"),
#         "label": SummaryWriter(r"logs\label"),
#          "pred": SummaryWriter(r"logs\pred"),
#         "lr": SummaryWriter(r"logs" + r"\lr")
#
#     }
#     TrainLoader = DataLoader(r"D:\Project\CUMT_PAPER_DATASETS_FINAL")
#     ValidLoader = DataLoader(r"D:\Github_Repo\Deploy")
#
#     #优化器
#     optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
#     # optimizer = optim.AdamW(net.parameters(), lr=lr)
#
#     # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer,
#     #                                               mode='triangular2',
#     #                                               max_lr=lr,
#     #                                               base_lr=1e-7,
#     #                                               step_size_up=step_size_up,
#     #                                               step_size_down=1,
#     #                                               cycle_momentum=False)
#     # scheduler =  torch.optim.lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)
#     #损失函数
#     # todo:change the loss function
#     best_loss = float("inf")
#     net.train()
#     #训练epochs
#     step = 0
#     for epoch in range(epochs):
# ########################################################################################################################
#         optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] * eval(input("输入"+str(epoch+1)+"学习率倍率："))
#         # if epochs // 3 >= epoch > epochs // 4:
#         #     batch_size = 2
#         # elif epoch > epochs // 3:
#         #     batch_size = 1
#         train_data = torch.utils.data.DataLoader(
#             dataset=TrainLoader,
#             batch_size=batch_size,
#             shuffle=True
#         )
#         # todo:
#         # if epoch == epochs // 2:
#         #     for param in net.backbone.parameters():
#         #         param.requires_grad = True
#         #     for param in net.decode_head.parameters():
#         #         param.requires_grad = True
#         if epoch < step_size_up:
#             optimizer.param_groups[0]['lr'] = lr
#         if epoch > epochs * 4 //5 and optimizer.param_groups[0]['lr'] > 1e-6:
#             optimizer.param_groups[0]['lr'] = 1e-6
#     ########################################################################################################################
#         for image, texture, label in train_data:
#             optimizer.zero_grad()
#
#             # 将数据拷贝到device中
#             #todo:
#             # image   = torch.cat([image,texture],dim=1).to(device=device, dtype=torch.float32)
#             image   = image.to(device=device, dtype=torch.float32)
#             texture = texture.to(device=device, dtype=torch.float32)
#             label   = label.to(device=device, dtype=torch.float32)
#
#             #预测图像
#             #todo:
#             pred = net(image,texture)
#             # pred = net(image)
#             # writer["pred"].add_images("pred", texture, step)
#             # writer["label"].add_images("label", label, step)
#
#             # todo:change the loss function
#             L1 = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(9).to(device))
#             # L1 = nn.BCEWithLogitsLoss()
#             loss = L1(pred, label)
#             # 保存loss值最小的网络参数
#             if loss < best_loss:
#                 best_loss = loss
#                 torch.save(net.state_dict(), r'model.pth')
#             # 更新参数
#             loss.backward()
#             optimizer.step()
#             step += 1
#             if step % 50 == 0:
#                 net.eval()
#                 with torch.no_grad():
#                     num_choose = random.randint(0, len(ValidLoader) - 1)
#                     val_image, val_texture, val_label = ValidLoader[num_choose]
#
#                     # val_image, val_texture, val_label = ValidLoader[step%3]
#
#                     #todo
#                     # val_image   = torch.cat([val_image,val_texture],dim=0).to(device=device, dtype=torch.float32)
#                     val_image = val_image.to(device=device, dtype=torch.float32)
#                     val_texture = val_texture.to(device=device, dtype=torch.float32)
#                     val_label   = val_label.to(device=device, dtype=torch.float32)
#
#                     val_image   = val_image.reshape(1, 4, val_image.shape[1], val_image.shape[2])
#                     val_texture = val_texture.reshape(1, 1, val_texture.shape[1], val_texture.shape[2])
#                     val_label   = val_label.reshape(1, 1, val_label.shape[1], val_label.shape[2])
#                     #todo:
#                     val_pred = net(val_image,val_texture)
#                     # val_pred = net(val_image)
#
#
#                     val_pred = torch.sigmoid(val_pred)
#
#                     val_pred[val_pred >  0.5] = 1
#                     val_pred[val_pred <= 0.5] = 0
#
#                     val_pred = np.array(val_pred.data.cpu())
#                     val_label = np.array(val_label.data.cpu())
#
#                     val_pred[np.isnan(val_pred)] = 0
#
#
#                     writer["pred"].add_images("pred", val_pred, step)
#                     writer["label"].add_images("label", val_label, step)
#
#                     val_label = val_label.reshape(-1)
#                     val_pred = val_pred.reshape(-1)
#
#                     recall = recall_score(val_label, val_pred)
#                     precision = precision_score(val_label, val_pred)
#                 net.train()
#                 writer['recall'].add_scalar("data", recall, step)
#                 writer['precision'].add_scalar("data", precision, step)
#             writer['loss'].add_scalar("data", loss, step)
#             writer['lr'].add_scalar("data", optimizer.state_dict()['param_groups'][0]['lr'], step)
#         # scheduler.step()
#
#     writer["loss"].close()
#     writer["recall"].close()
#     writer["precision"].close()
#     writer["label"].close()
#     writer["pred"].close()
#     writer["lr"].close()
class TrainingStrategy:
    def __init__(self,
                 device,
                 net,
                 batch_size,
                 epochs,
                 optimizer,
                 learning_rate,
                 loss,
                 backbones,
                 training_data_path,
                 validate_data_path = None,
                 data_distributor   = "random",
                 logs_path = None
                 ):
        self.device             = device
        self.net                = net
        self.batch_size         = batch_size
        self.epochs             = epochs
        self.optimizer          = optimizer
        self.learning_rate      = learning_rate
        self.loss               = loss
        self.backbones          = backbones
        self.training_data_path = training_data_path
        self.validate_data_path = validate_data_path
        self.distributor        = data_distributor
        self.logs_path          = logs_path
        self.validate_mode      = "random"
        self.create_logs()
    def create_logs(self):
        print("准备生成日志文件目录")
        if self.logs_path is None:
            self.logs_path = r"logs"
        writer = {
            "loss"      : SummaryWriter(self.logs_path  + r"\loss"),
            'recall'    : SummaryWriter(self.logs_path  + r"\recall"),
            'precision' : SummaryWriter(self.logs_path  + r"\precision"),
            "label"     : SummaryWriter(self.logs_path  + r"\label"),
            "pred"      : SummaryWriter(self.logs_path  + r"\pred"),
            "lr"        : SummaryWriter(self.logs_path  + r"\lr")
        }
        print("日志文件目录生成完成")
    def train(self):
        step = 0
########################################################################################################################
        # 如果没有验证样本，验证集从训练样本中随机抽取（如分配比：0.1即从总样本中抽取10%作为验证集）
        # 如果有验证样本，直接读取验证样本路径
        print("开始读取验证数据")
        if self.validate_data_path is not None:
            T, V = Alloted_Distributor.Distributor(self.training_data_path, self.validate_data_path)
            print("(指定路径读取验证)")
        else:
            if self.distributor == "random":
                distribute_rate = eval(input("输入随机分配验证集的比率(0.0-1.0):"))
                while type(distribute_rate) != float:
                    distribute_rate = eval(input("重新输入随机分配验证集的比率(0.0-1.0):"))
                T,V = Random_Distributor.Distributor(self.training_data_path,distribute_rate)
                print("(总样本中随机抽取"+str(distribute_rate)+"验证)")
            else:
                N = eval(input("输入交叉验证折数:"))
                while type(N) != float:
                    distribute_rate = eval(input("重新输入交叉验证折数:"))
                T,V = N_Cross_Distributor.Distributor(self.training_data_path,N)
                print("(总样本种进行"+str(N)+"折交叉验证)")
        ValidLoader = DataLoader(V)
        print("训练数据读取完成")
        print("开始读取训练数据")
        TrainLoader = DataLoader(T)
        print("训练数据读取完成")

        TrainData = torch.utils.data.DataLoader(
            dataset=TrainLoader,
            batch_size=self.batch_size,
            shuffle=True
        )
        ValidateData = torch.utils.data.DataLoader(
            dataset=ValidLoader,
            batch_size=self.batch_size,
            shuffle=True
        )

        BestLoss = float("inf")
        self.net.train()

        for epoch in range(self.epochs):
            for image, label in TrainData:
                self.optimizer.zero_grad()

                image = image.to(device=self.device, dtype=torch.float32)
                label = label.to(device=self.device, dtype=torch.float32)

                prediction = self.net(image, label)
                L1 = nn.BCEWithLogitsLoss()
                Loss = L1(prediction, label)

                if Loss < BestLoss:
                    BestLoss = Loss
                    torch.save(self.net.state_dict(), "model.pth")

                Loss.backward()
                self.optimizer.step()
                step += 1

            self.net.eval()
    #         with torch.no_grad():
    #             for val_image1, val_image2, val_label in ValidLoader
    #             # "single" 即每隔一个epoch单样本跟踪验证
    #             if self.validate_mode == "single":
    #                 val_image, val_texture, val_label = ValidLoader[step%len(ValidLoader)]
    #             # "batch" 即每隔一个epoch批量样本跟踪验证，最后统一计算精度指标
    #             if self.validate_mode == "batch":
    #
    #                 image = image.to(device=self.device, dtype=torch.float32)
    #                 label = label.to(device=self.device, dtype=torch.float32)
    #
    #                 val_image, val_texture, val_label = ValidLoader[step%len(ValidLoader)]
    #             # todo
    #             # val_image   = torch.cat([val_image,val_texture],dim=0).to(device=device, dtype=torch.float32)
    #             val_image = val_image.to(device=device, dtype=torch.float32)
    #             val_texture = val_texture.to(device=device, dtype=torch.float32)
    #             val_label = val_label.to(device=device, dtype=torch.float32)
    #
    #             val_image = val_image.reshape(1, 4, val_image.shape[1], val_image.shape[2])
    #             val_texture = val_texture.reshape(1, 1, val_texture.shape[1], val_texture.shape[2])
    #             val_label = val_label.reshape(1, 1, val_label.shape[1], val_label.shape[2])
    #             # todo:
    #             val_pred = net(val_image, val_texture)
    #             # val_pred = net(val_image)
    #
    #             val_pred = torch.sigmoid(val_pred)
    #
    #             val_pred[val_pred > 0.5] = 1
    #             val_pred[val_pred <= 0.5] = 0
    #
    #             val_pred = np.array(val_pred.data.cpu())
    #             val_label = np.array(val_label.data.cpu())
    #
    #             val_pred[np.isnan(val_pred)] = 0
    #
    #             writer["pred"].add_images("pred", val_pred, step)
    #             writer["label"].add_images("label", val_label, step)
    #
    #             val_label = val_label.reshape(-1)
    #             val_pred = val_pred.reshape(-1)
    #
    #             recall = recall_score(val_label, val_pred)
    #             precision = precision_score(val_label, val_pred)
    #         net.train()
    #         writer['recall'].add_scalar("data", recall, step)
    #         writer['precision'].add_scalar("data", precision, step)
    #     writer['loss'].add_scalar("data", loss, step)
    #     writer['lr'].add_scalar("data", optimizer.state_dict()['param_groups'][0]['lr'], step)
    #     # scheduler.step()
    #
    # writer["loss"].close()
    # writer["recall"].close()
    # writer["precision"].close()
    # writer["label"].close()
    # writer["pred"].close()
    # writer["lr"].close()



if __name__=="__main__":
    TrainingStrategy(
        device='cuda',
        net=SegFormerUNetConcise(),
        batch_size=4,
        epochs=10,
        optimizer=None,
        learning_rate=None,
        loss=None,
        backbones=None,
        training_data_path="D:\Project\CUMT_PAPER_DATASETS",
        validate_data_path=None,
        data_distributor="random",
        logs_path=None
    ).train()