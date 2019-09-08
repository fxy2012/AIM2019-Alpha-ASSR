'''
	flops: 654.9 G
	params: 7442328
'''


import torch
import torch.nn as nn
from collections import OrderedDict


class MeanShift(nn.Conv2d):
    def __init__(
        self, rgb_range,
        rgb_mean=(0.4488, 0.4371, 0.4040), rgb_std=(1.0, 1.0, 1.0), sign=-1):

        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


## Channel Attention (CA) Layer
class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()
        # global average pooling: feature --> point
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
                nn.Conv2d(channel, channel // reduction, 1, padding=0, bias=True),
                nn.ReLU(inplace=True),
                nn.Conv2d(channel // reduction, channel, 1, padding=0, bias=True),
                nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


class one_conv(nn.Module):
    def __init__(self, inchanels, growth_rate, kernel_size=3):
        super(one_conv,self).__init__()
        self.conv = nn.Conv2d(inchanels, growth_rate, kernel_size=kernel_size, padding=kernel_size // 2, stride=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        output = self.relu(self.conv(x))
        return torch.cat((x, output), 1)


class RDB(nn.Module):
    def __init__(self, G0=64, C=6, G=32, kernel_size = 3):
        super(RDB,self).__init__()
        self.conv1 = nn.Conv2d(G0, G, kernel_size, 1, kernel_size // 2, bias=False)
        self.conv2 = nn.Conv2d(G, G0, kernel_size, 1, kernel_size // 2, bias=False)
        self.agg_conv = nn.Conv2d(G+G0, G0, 3, 1, 3//2, bias=False)
        self.act = nn.ReLU(True)
    def forward(self, x):
        y1 = self.act(self.conv1(x))
        y2 = self.act(self.conv2(y1))
        inp = torch.cat((y1, y2), 1)
        out = self.act(self.agg_conv(inp))
        return out + x


class Agg_unit(nn.Module):
    def __init__(self, G0=64, C=6, G=32, kernel_size=3):
        super(Agg_unit, self).__init__()
        self.RDB1 = RDB(G0=G0, C=C, G=G, kernel_size=kernel_size)
        self.RDB2 = RDB(G0=G0, C=C, G=G, kernel_size=kernel_size)
        self.RDB3 = RDB(G0=G0, C=C, G=G, kernel_size=kernel_size)
        self.RDB4 = RDB(G0=G0, C=C, G=G, kernel_size=kernel_size)
        #self.conv1 = make_layer_s(in_feats=2*G0, out_feats=G0, stride=1)
        #self.conv2 = make_layer_s(in_feats=3*G0, out_feats=G0, stride=1)
        self.conv1 = nn.Conv2d(2*G0, G0, 3, 1, 1)
    def forward(self, x):
        y1 = self.RDB1(x)
        y2 = self.RDB2(y1)
        y3 = self.RDB3(y2)
        y4 = self.RDB4(y3)
        inp2 = torch.cat((y1, y4), 1)
        out = self.conv1(inp2)
        #out_cat = torch.cat((first, out), 1)
        return y4, out


class Agg(nn.Module):
    def __init__(self, unit_num, G0=64, C=6, G=32, kernel_size=3, res_scale=0.1):
        super(Agg, self).__init__()
        self.unit_num = unit_num
        self.res_scale = res_scale
        self.unit = nn.ModuleList()
        for i in range(self.unit_num):
            self.unit.append(Agg_unit(G0=G0, C=C, G=G, kernel_size=kernel_size))
        self.conv = nn.Conv2d(unit_num*G0, G0, 1, 1, 1 // 2, bias=False)
        self.conv2 = nn.Conv2d(G0, G, 3, 1, 3 // 2, bias=False)
        #self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        y, y_up = self.unit[0](x)
        y1 = y_up
        for i in range(self.unit_num-1):
            y, y_up = self.unit[i+1](y)
            y1 = torch.cat((y1, y_up), 1)
        out = self.conv2(self.conv(y1)) 
        #return x + self.res_scale*out
        return out


def make_model(args, parent=False):
    return ASSR()

class ASSR(nn.Module):
    def __init__(self, scale=4):
        super(ASSR, self).__init__()
        self.unit_num = 4
        self.scale = scale
        self.n_color = 3
        self.G = 40
        self.G0 = 40
        self.C = 2
        self.kernel_size = 3
        self.res_scale = 0.1
        self.sub_mean = MeanShift(255)
        self.add_mean = MeanShift(255, sign=1)
        head = []
        head.append(
            nn.Conv2d(self.n_color, self.G, self.kernel_size, 1, self.kernel_size // 2, bias=False)
        )
        body = []
        body.append(nn.Conv2d(self.G, self.G0, self.kernel_size, 1, self.kernel_size // 2, bias=False))
        body.append(
            Agg(self.unit_num, G0=self.G0, C=self.C, G=self.G, kernel_size=self.kernel_size, res_scale=self.res_scale)
        )
        body.append(CALayer(40, 4))
        tail = []
        tail.append(
            nn.Conv2d(self.G, self.G*self.scale*self.scale, self.kernel_size, 1, self.kernel_size // 2, bias=False)
        )
        tail.append(
            nn.PixelShuffle(self.scale)
        )
        tail.append(
            nn.Conv2d(self.G, self.n_color, self.kernel_size, 1, self.kernel_size // 2,
                      bias=False)
        )
        self.head = nn.Sequential(*head)
        self.body = nn.Sequential(*body)
        self.tail = nn.Sequential(*tail)

    def forward(self, x):

        y1 = self.head(self.sub_mean(x))
        y = self.body(y1)
        y = y + y1
        y = self.tail(y)
        y = self.add_mean(y)

        return y


    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))














