# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA Corporation. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# ---------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F

from .backbone import mit_b0, mit_b1, mit_b2, mit_b3, mit_b4, mit_b5


class MLP(nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x
    
class ConvModule(nn.Module):
    def __init__(self, c1, c2, k=1, s=1, p=0, g=1, act=True):
        super(ConvModule, self).__init__()
        self.conv   = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn     = nn.BatchNorm2d(c2, eps=0.001, momentum=0.03)
        self.act    = nn.ReLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))

class Decoder(nn.Module):
    def __init__(self, in_channels=[32, 64, 160, 256], embedding_dim=768, dropout_ratio=0.1):
        super(Decoder, self).__init__()
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels

        self.linear_c4 = MLP(input_dim=c4_in_channels, embed_dim=embedding_dim)
        self.linear_c3 = MLP(input_dim=c3_in_channels, embed_dim=embedding_dim)
        self.linear_c2 = MLP(input_dim=c2_in_channels, embed_dim=embedding_dim)
        self.linear_c1 = MLP(input_dim=c1_in_channels, embed_dim=embedding_dim)

        self.linear_fuse = ConvModule(
            c1=embedding_dim * 4,
            c2=embedding_dim,
            k=1,
        )

        self.linear_pred = nn.ConvTranspose2d(embedding_dim, 64, kernel_size=2, stride=2)
        self.dropout = nn.Dropout2d(dropout_ratio)

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4).permute(0, 2, 1).reshape(n, -1, c4.shape[2], c4.shape[3])
        _c4 = F.interpolate(_c4, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c3 = self.linear_c3(c3).permute(0, 2, 1).reshape(n, -1, c3.shape[2], c3.shape[3])
        _c3 = F.interpolate(_c3, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c2 = self.linear_c2(c2).permute(0, 2, 1).reshape(n, -1, c2.shape[2], c2.shape[3])
        _c2 = F.interpolate(_c2, size=c1.size()[2:], mode='bilinear', align_corners=False)

        _c1 = self.linear_c1(c1).permute(0, 2, 1).reshape(n, -1, c1.shape[2], c1.shape[3])

        _c = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))
        x = self.dropout(_c)
        x = self.linear_pred(x)
        return x

#2次卷积#################################################################################################################
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.double_conv(x)

#下采样##################################################################################################################
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.down = nn.Sequential(
            # nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels, in_channels, kernel_size=2, stride=2, padding=0),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.down(x)
#上采样##################################################################################################################
class UpFromSem(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up_DoubleConv= DoubleConv(in_channels, out_channels)
    def forward(self, x_down, x_skip_connection):
        x_forward = torch.concat([x_down,x_skip_connection], dim=1)
        return self.up_DoubleConv(x_forward)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Sequential(
            # nn.Upsample(scale_factor=2,mode="bilinear",align_corners=True)
            nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        )
        self.up_DoubleConv= DoubleConv(in_channels, out_channels)

    def forward(self, x_down, x_skip_connection):
        x_up = self.up(x_down)
        x_forward = torch.concat([x_up,x_skip_connection], dim=1)
        return self.up_DoubleConv(x_forward)

#输出层##################################################################################################################
class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self,x):
        return self.conv(x)

class SegFormerUNetConcise(nn.Module):
    def __init__(self, in_channels = 3, num_classes=21, backbone='b0', pretrained=False):
        super(SegFormerUNetConcise, self).__init__()
        self.num_classes = num_classes
        self.in_channel = in_channels
        self.in_channels = {
            'b0': [32, 64, 160, 256], 'b1': [64, 128, 320, 512], 'b2': [64, 128, 320, 512],
            'b3': [64, 128, 320, 512], 'b4': [64, 128, 320, 512], 'b5': [64, 128, 320, 512],
        }[backbone]
        self.backbone   = {
            'b0': mit_b0, 'b1': mit_b1, 'b2': mit_b2,
            'b3': mit_b3, 'b4': mit_b4, 'b5': mit_b5,
        }[backbone](pretrained,in_channels)
        self.embedding_dim   = {
            'b0': 256, 'b1': 256, 'b2': 768,
            'b3': 768, 'b4': 768, 'b5': 768,
        }[backbone]
        self.decode_head = Decoder(self.in_channels, self.embedding_dim)

        self.input_image = DoubleConv(1, 32)
        self.Down1 = Down(32, 64)
        self.Up1   = UpFromSem(128, 64)
        self.Up2   = Up(64, 32)
        self.output_layer = OutConv(32, self.num_classes)


    def forward(self, inputs, sematic_info):

        x = self.backbone.forward(inputs)
        x = self.decode_head.forward(x)

        x_512 = self.input_image(sematic_info)
        x_256 = self.Down1(x_512)

        x_256 = self.Up1(x, x_256)
        x_512 = self.Up2(x_256, x_512)

        x = self.output_layer(x_512)

        # x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x
