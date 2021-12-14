import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from .Shift3D import Shift3DLayer
from .utils import * 

class Fire(nn.Module):

    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes,
                 use_bypass=False):
        super(Fire, self).__init__()
        self.use_bypass = use_bypass
        self.inplanes = inplanes
        self.relu = nn.ReLU(inplace=True)
        self.squeeze = nn.Conv3d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_bn = nn.BatchNorm3d(squeeze_planes, eps=0.001, momentum=0.01)
        self.expand1x1 = nn.Conv3d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_bn = nn.BatchNorm3d(expand1x1_planes, eps=0.001, momentum=0.01)
        self.expand3x3 = nn.Conv3d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_bn = nn.BatchNorm3d(expand3x3_planes, eps=0.001, momentum=0.01)

    def forward(self, x):
        out = self.squeeze(x)
        out = self.squeeze_bn(out)
        out = self.relu(out)

        out1 = self.expand1x1(out)
        out1 = self.expand1x1_bn(out1)
        
        out2 = self.expand3x3(out)
        out2 = self.expand3x3_bn(out2)

        out = torch.cat([out1, out2], 1)
        if self.use_bypass:
            out += x
        out = self.relu(out)

        return out

# SqueezeNet3D or ShiftNet3D (SqueezeNet3D + Shift3D)
class SqueezeNet3D(nn.Module):
    def __init__(self, shift3d=True, shift_chance=0.25, batch_shift=False, decay_iterations = 0, squeezec = 32, input_channel=1, 
                 sample_size = WIDTH, sample_depth = DEPTH,num_classes=2, use_classifier=True):
        super(SqueezeNet3D, self).__init__()
        self.num_classes = num_classes
        self.use_classifier = use_classifier
        last_duration = int(math.ceil(sample_depth / 32))
        last_size = int(math.ceil(sample_size / 32))
        self.shift3d = shift3d

        if shift3d:
            self.shift3d_layer = Shift3DLayer(shift_chance=shift_chance, batch_shift=batch_shift, decay_iterations=decay_iterations)
        self.convbn = nn.Sequential(
            nn.Conv3d(input_channel, 64, kernel_size=3, stride=2, padding=(1,1,1)),
            nn.BatchNorm3d(64, eps=0.001, momentum=0.01),
             # make sure inplace for this relu is False, 
            # otherwise it may affects Shift3D operation in Pytorch version 1.2 or later
            nn.ReLU(inplace=False)
        )
        
        self.features = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(64, squeezec, 64, 64),
            Fire(128, squeezec, 64, 64, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(128, squeezec*2, 128, 128),
            Fire(256, squeezec*2, 128, 128, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(256, squeezec*3, 192, 192),
            Fire(384, squeezec*3, 192, 192, use_bypass=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1),
            Fire(384, squeezec*4, 256, 256),
            Fire(512, squeezec*4, 256, 256, use_bypass=True),
        )
        
        if use_classifier:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Conv3d(512, self.num_classes, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
            )

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x):
        x = self.convbn(x)
        if self.shift3d and self.training:
            x = self.shift3d_layer(x)
        x = self.features(x)
        
        if self.use_classifier:
            x = self.classifier(x)
            return x.view(x.size(0), -1)
        else:
            return x