import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from .Shift3D import Shift3DLayer
from .utils import * 


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes, momentum = 0.01)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes, momentum = 0.01)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

# https://github.com/okankop/Efficient-3DCNNs/blob/master/models/resnet.py
# Revised to accept Shift3D layer
class ResNet3D(nn.Module):

    def __init__(self,block, layers, shift3d=False, shift_chance=0.25, batch_shift=False, decay_iterations = 0, sample_size = WIDTH,
                 sample_duration = DEPTH, shortcut_type='B', num_classes=2):
        self.inplanes = 64
        super(ResNet3D, self).__init__()
        self.conv1 = nn.Conv3d(1,64,kernel_size=5,stride=(2, 2, 2),padding=(2, 2, 2),bias=False)
        self.bn1 = nn.BatchNorm3d(64, momentum = 0.01)
        self.relu = nn.ReLU(inplace=False) #inplace must be false for shift3d

        self.shift3d = shift3d
        if shift3d:
            self.shift3d_layer = Shift3DLayer(shift_chance=shift_chance, batch_shift=batch_shift, decay_iterations=decay_iterations)

        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type, stride=(2, 2, 2))
        self.layer2 = self._make_layer(block, 128, layers[1], shortcut_type, stride=(2, 2, 2))
        self.layer3 = self._make_layer(block, 256, layers[2], shortcut_type, stride=(2, 2, 2))
        self.layer4 = self._make_layer(block, 512, layers[3], shortcut_type, stride=(2, 2, 2))
        
        last_duration = int(math.ceil(sample_duration / 64))
        last_size = int(math.ceil(sample_size / 64))
        self.avgpool = nn.AvgPool3d((last_duration, last_size, last_size), stride=1)

        self.fc = nn.Linear(512, num_classes)
    
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight)
#                 m.weight.data.fill_(1)
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes, momentum = 0.01))

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        if self.shift3d and self.training:
            x = self.shift3d_layer(x)

        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def resnet3d34(**kwargs):
    """Constructs a ResNet-34 3D model.
    """
    model = ResNet3D(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model