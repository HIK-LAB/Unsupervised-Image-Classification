import torch
import torch.nn as nn
import os
import random
import math
from torch.nn import functional as F


__all__ = ['resnet50']

class GBlur(nn.Module):
    """ A GPU implemented gaussian bluring data augmentation during training"""
    def __init__(self, kernel_size=23, channel=3, sigma_range=(0.1, 2.0), dim=2):
        super(GBlur, self).__init__()
        self.channel = channel
        self.dim = dim
        self.pad = int(kernel_size / 2)
        self.sigma_range = sigma_range

        self.kernel_size = [kernel_size] * dim
        self.meshgrids = torch.meshgrid(
            [
                torch.arange(size, dtype=torch.float32)
                for size in self.kernel_size
            ]
        )
        
    def forward(self, x, p=0.5):
        if random.random() > p:
            return x
        sigma = random.random() * (self.sigma_range[1] - self.sigma_range[0]) + self.sigma_range[0]
        self.sigma = [sigma] * self.dim
        kernel = 1
        for size, std, mgrid in zip(self.kernel_size, self.sigma, self.meshgrids):
            mean = (size - 1) / 2
            kernel *= 1 / (std * math.sqrt(2 * math.pi)) * \
                      torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(self.channel, *[1] * (kernel.dim() - 1)).cuda()
        x = F.conv2d(x, weight=kernel, groups=self.channel, padding=self.pad)
        return x

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        # if dilation > 1:
        #     raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        if isinstance(dilation, int):
            dilation1 = dilation2 = dilation
        else:
            assert isinstance(dilation, (list, tuple)) and len(dilation) == 2
            dilation1, dilation2 = dilation

        self.conv1 = conv3x3(inplanes, planes, stride, dilation=dilation1)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, dilation=dilation2)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, dim=3, extra_mlp=False, gblur=False, 
                 linear_eval=False, width_mult=1, input_size=224):
        super(ResNet, self).__init__()
        # settings
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.extra_mlp = extra_mlp
        self.linear_eval = linear_eval
        self.inplanes = 64 * width_mult
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group

        # network backbone
        self.conv1 = nn.Conv2d(dim, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64 * width_mult, layers[0])
        self.layer2 = self._make_layer(block, 128 * width_mult, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256 * width_mult, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512 * width_mult, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # linear layer
        # extra mlp layer is utilized to further boost the performance
        if extra_mlp:
            self.head = nn.Sequential(conv1x1(512 * width_mult * block.expansion, 2048),
                norm_layer(2048),
                nn.ReLU(True),
                conv1x1(2048, 2048),
                norm_layer(2048),
                nn.ReLU(True)
                )
            self.fc = nn.Linear(2048, num_classes)
        else:
            self.fc = nn.Linear(512 * width_mult * block.expansion, num_classes)
        if self.linear_eval:
            self.linear = nn.Linear(512 * width_mult * block.expansion, num_classes)

        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

        #random gaussian bluring
        if gblur:
            self.gblur = GBlur(kernel_size=int((input_size * 0.1) // 2 + 1), channel=dim)
        else:
            self.gblur = None

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        first_block_dilation = previous_dilation if isinstance(block, Bottleneck) else (previous_dilation, self.dilation)
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, first_block_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def forward(self, x):
        if not self.linear_eval:
            with torch.no_grad():
                if self.gblur:
                    x = self.gblur(x)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x = self.maxpool(x)

            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            if self.extra_mlp:
                x = self.head(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        else:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.layer4(x)
                x = self.avgpool(x)
                if self.extra_mlp:
                    x = self.head(x)
                x = torch.flatten(x, 1)
                x = x.detach()
            x = self.linear(x)

        return x

def _resnet(block, layers, out=1000, dim=3, **kwargs):
    model = ResNet(block, layers, num_classes=out, dim=dim, **kwargs)
    return model

def resnet50(out=1000, extra_mlp=False, random_gblur=False, linear_eval=False, dim=3, **kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], out=out, dim=dim, \
        extra_mlp=extra_mlp, gblur=random_gblur, linear_eval=linear_eval, \
        width_mult=1, **kwargs)
