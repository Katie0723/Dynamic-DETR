import torch
import torch.nn as  nn
import torch.nn.functional as F
import numpy as np
from torchvision.ops import deform_conv2d
from .dyrelu import DyReLUB
import copy

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        s, b, c, _, _ = x.size()
        y = self.avg_pool(x).view(s, b, c)
        y = self.fc(y).view(s, b, c, 1, 1)
        return x * y.expand_as(x)

class DeformConv(nn.Module):
  def __init__(self, in_channels=256, out_channels=64, kernel_size=3, stride=1, padding=1):
    super(DeformConv, self).__init__()
    self.stride = stride
    self.padding = padding
    self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
    self.conv_offset = nn.Conv2d(in_channels, 2*kernel_size*kernel_size, kernel_size=kernel_size, stride=stride, padding=padding)
    init_offset = torch.Tensor(np.zeros([2*kernel_size*kernel_size, in_channels, kernel_size, kernel_size]))
    self.conv_offset.weight = torch.nn.Parameter(init_offset)
    self.batchnorm = nn.BatchNorm2d(in_channels)
    #self.layer_norm = nn.LayerNorm(256)
    #layer_norm = nn.LayerNorm([C, H, W])
    #>>> output = layer_norm(input)
  def forward(self, x):
    offset = self.conv_offset(x)
    out = deform_conv2d(input=x, offset=offset, weight=self.conv.weight, stride=(self.stride,self.stride), padding=(self.padding, self.padding))
    out = self.batchnorm(out)
    #print(out.shape)
    #layer_norm = nn.LayerNorm([int(out.shape[0]), int(out.shape[1]), int(out.shape[2])])
    #out = layer_norm(out)
    #print('after: '+str(out.shape))
    return out

class EncoderLayer(nn.Module):

    def __init__(self):
        super().__init__()
        DeformConvLayer = DeformConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.deform_conv2d = _get_clones(DeformConvLayer, 5)

        DownsamplingLayer  = DeformConv(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.Downsampling = _get_clones(DownsamplingLayer, 4)
        UpsamplingLayer = nn.Sequential(
                DeformConv(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
                nn.UpsamplingBilinear2d(scale_factor=2)
        )
        self.Upsampling = _get_clones(UpsamplingLayer, 4)
        
        self.batchnorm0 = nn.BatchNorm2d(256)
        self.batchnorm1 = nn.BatchNorm2d(256)
        self.batchnorm2 = nn.BatchNorm2d(256)
        self.batchnorm3 = nn.BatchNorm2d(256)
        self.batchnorm4 = nn.BatchNorm2d(256)
        SE = SELayer(256,reduction=16)
        self.SE = _get_clones(SE, 5)

        self.dyrelub = DyReLUB(channels=256, conv_type='2d')
    def forward(self, p):
        
        #print('A: \n'+str(out3[0]))
        out2 = self.deform_conv2d[0](p[0]) # p2 largest
        out3 = self.deform_conv2d[1](p[1])
        out4 = self.deform_conv2d[2](p[2])
        out5 = self.deform_conv2d[3](p[3])
        out6 = self.deform_conv2d[4](p[4]) # p6 smallest 
        
        out_up2 = self.Upsampling[0](p[1]) 
        #print('after: '+str(out_up2))
        out_up3 = self.Upsampling[1](p[2])
        out_up4 = self.Upsampling[2](p[3])
        out_up5 = self.Upsampling[3](p[4])
        
        out_down3 = self.Downsampling[0](p[0])
        out_down4 = self.Downsampling[1](p[1])
        out_down5 = self.Downsampling[2](p[2])
        out_down6 = self.Downsampling[3](p[3])
        #c = torch.stack((out2,out_up2))
        
        p2plus = self.SE[0](torch.stack((out2,out_up2)))
        p3plus = self.SE[1](torch.stack((out3,out_up3,out_down3)))
        p4plus = self.SE[2](torch.stack((out4,out_up4,out_down4)))
        p5plus = self.SE[3](torch.stack((out5,out_up5,out_down5)))
        p6plus = self.SE[4](torch.stack((out6,out_down6)))
        
        out_layer0 = self.dyrelub(self.batchnorm0(torch.sum(p2plus,0)))
        out_layer1 = self.dyrelub(self.batchnorm1(torch.sum(p3plus,0)))
        out_layer2 = self.dyrelub(self.batchnorm2(torch.sum(p4plus,0)))
        out_layer3 = self.dyrelub(self.batchnorm3(torch.sum(p5plus,0)))
        out_layer4 = self.dyrelub(self.batchnorm4(torch.sum(p6plus,0)))

        out_layer = [out_layer0, out_layer1, out_layer2, out_layer3, out_layer4]
        #for i in range(5):
        #  print(out_layer[i].shape)

        return out_layer
        
class Encoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        #self.batchnorm = nn.BatchNorm2d(256, affine=False)
    def forward(self, p2, p3, p4, p5, p6):
        out_p = [p2, p3, p4, p5, p6]
        cnt=0
        for layer in self.layers:
            out_p = layer(out_p)
            #print('out_layer{}: '.format(cnt)+str(out_p[3]))
            #print('==================================================================')
            cnt+=1
        #out_p[3] = self.batchnorm(out_p[3])
        return out_p