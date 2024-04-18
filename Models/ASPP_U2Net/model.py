from Models.ASPP_U2Net.parts import *
# from parts import *

class ASPPU2Net(nn.Module):
    def __init__(self, image_channels, texture_channels,num_classes):
        super().__init__()
        self.image_channels = image_channels
        self.texture_channels = texture_channels
        self.classes = num_classes
########Encoder
        self.input_image = DoubleConv(self.image_channels, 32)
        self.input_texture = DoubleConv(self.texture_channels, 32)

        self.Down1 = Down(32,64)
        self.Down2 = Down(64,128)
        self.Down3 = Down(128,256)
        self.Down4 = Down(256, 512)

########Decoder
        self.Up1 = Up(1024,512)
        self.Up2 = Up(512,256)
        self.Up3 = Up(256,128)
        self.Up4 = Up(128,64)
        self.output_layer = OutConv(64, self.classes)

        self.ASPP1 = ASPP(512, 512)
        self.ASPP2 = ASPP(256, 256)
        self.ASPP3 = ASPP(128, 128)
        self.ASPP4 = ASPP(64, 64)


    def forward(self, image, texture):
########Encoder
        ori_1 = self.input_image(image)
        tex_1 = self.input_texture(texture)
        x1 = torch.concat([ori_1, tex_1], dim=1)

        ori_2 = self.Down1(ori_1)
        tex_2 = self.Down1(tex_1)
        x2 = torch.concat([ori_2, tex_2], dim=1)

        ori_3 = self.Down2(ori_2)
        tex_3 = self.Down2(tex_2)
        x3 = torch.concat([ori_3, tex_3], dim=1)

        ori_4 = self.Down3(ori_3)
        tex_4 = self.Down3(tex_3)
        x4 = torch.concat([ori_4, tex_4],dim=1)

        ori_5 = self.Down4(ori_4)
        tex_5 = self.Down4(tex_4)
        x5 = torch.concat([ori_5, tex_5], dim=1)

########Decoder
        x4 = self.ASPP1(x4)
        x3 = self.ASPP2(x3)
        x2 = self.ASPP3(x2)
        x1 = self.ASPP4(x1)
        x = self.Up1(x5,x4)
        x = self.Up2(x, x3)
        x = self.Up3(x, x2)
        x = self.Up4(x, x1)


        logits = self.output_layer(x)
        return logits

if __name__ == "__main__":
    net = ASPPU2Net(image_channels=4, texture_channels=1, classes=1)
    print(net)

