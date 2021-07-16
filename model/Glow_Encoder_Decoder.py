import torch
import torch.nn as nn
import torch.nn.functional as nnf
from torch.nn.modules import activation, conv
from model import common
from option import args
import numpy as np
# from guided_filter_pytorch.guided_filter import FastGuidedFilter, GuidedFilter, ConvGuidedFilter
from torchvision import models

def make_model(opt):
    print('make model glow')
    return Glow_Encoder_Decoder(opt)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.conv = nn.Conv2d((in_size+out_size), out_size, kernel_size=(3,3), padding=1)

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)
        x = self.conv(x)

        return x


class Glow_Encoder_Decoder(nn.Module):
    def __init__(self, opt, in_channels=3, out_channels=3):
        super(Glow_Encoder_Decoder, self).__init__()

        vgg = models.vgg16(pretrained=True)
        featureExtract = [module for module in vgg.features]
        new_layers = []
        for n, i in enumerate(featureExtract):
            new_layers.append(featureExtract[n])
        # self.featureExtract = nn.Sequential(*new_layers)

        self.conv1 = nn.Sequential(*new_layers[:5])
        self.conv2 = nn.Sequential(*new_layers[5:10])
        self.conv3 = nn.Sequential(*new_layers[10:17])
        self.conv4 = nn.Sequential(*new_layers[17:24])
        self.conv5 = nn.Sequential(*new_layers[24:])
        self.conv6 = nn.Sequential(*new_layers[24:])

        self.up1 = UNetUp(512, 256, dropout=0.5)
        self.up2 = UNetUp(256, 128, dropout=0.5)
        self.up3 = UNetUp(128, 64, dropout=0.5)
        self.up4 = UNetUp(64, 32, dropout=0.5)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, 2, 1, bias=False),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.Conv2d(8, 3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        # self.final = nn.Sequential(
        #     nn.Upsample(scale_factor=2),
        #     nn.ZeroPad2d((1, 0, 1, 0)),
        #     nn.Conv2d(128, out_channels, 4, padding=1),
        #     nn.Tanh(),
        # )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.conv1(x)
        # print(d1.size())
        d2 = self.conv2(d1)
        # print(d2.size())
        d3 = self.conv3(d2)
        # print(d3.size())
        d4 = self.conv4(d3)
        # print(d4.size())
        d5 = self.conv5(d4)
        # print(d5.size())
        #8 512 8 8
        # print('-------------')
        #8 512 16 16
        u1 = self.up1(d5, d4)
        # print(u1.size())
        u2 = self.up2(u1, d3)
        # print(u2.size())
        u3 = self.up3(u2, d2)
        # print(u3.size())
        u4 = self.up4(u3, d1)
        # print(u4.size())

        return self.final(u4)