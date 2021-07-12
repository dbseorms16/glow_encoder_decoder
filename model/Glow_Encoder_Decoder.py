import torch
import torch.nn as nn
import torch.nn.functional as nnf
from model import common
from option import args


def make_model(opt):
    print('make model glow')
    return Glow_Encoder_Decoder(opt)


class Glow_Encoder_Decoder(nn.Module):
    def __init__(self, opt, conv=common.default_conv):
        super(Glow_Encoder_Decoder, self).__init__()
        self.head = conv(opt.n_colors, 32, 3)
    
    def forward(self, x):
        x = self.head(x)
        
        results = []
        return results