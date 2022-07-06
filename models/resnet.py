import torch.nn as nn
import torch
import math
from .common import BasicBlock, Bottleneck

# Part from:
#     https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/utils.py

pre_trained_path = {'resnet18':"models/resnet_per_trained/resnet18-5c106cde.pth",
                    'resnet34':"models/resnet_per_trained/resnet34-333f7ec4.pth",
                    'resnet50':"models/resnet_per_trained/resnet50-19c8e357.pth",
                    'resnet101':"models/resnet_per_trained/resnet101-5d3b4d8f.pth",
                    'resnet152':"models/resnet_per_trained/resnet152-b121ed2d.pth"}

class ResNet(nn.Module):

    def __init__(self, num_classes, block, layers, yolo_use=True):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.yolo_use = yolo_use
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        if self.yolo_use:
            if block == Bottleneck:
                self.to_yolo_f3 = nn.Conv2d(512, 256, kernel_size=1, bias=False)
                self.to_yolo_f4 = nn.Conv2d(1024, 512, kernel_size=1, bias=False)
                self.to_yolo_f5 = nn.Conv2d(2048, 1024, kernel_size=1, bias=False)
            elif block == BasicBlock:
                self.to_yolo_f3 = nn.Conv2d(128, 256, kernel_size=1, bias=False)
                self.to_yolo_f4 = nn.Conv2d(256, 512, kernel_size=1, bias=False)
                self.to_yolo_f5 = nn.Conv2d(512, 1024, kernel_size=1, bias=False)
            else:
                raise NotImplementedError("unsupported block")

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.freeze_bn()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def yolo_extract(self, inputs):
        if self.yolo_use is False:
            raise RuntimeError("This Resnet backbone setting is not ready for yolo, change your settings!")
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x2 = self.to_yolo_f3(x2)
        x3 = self.to_yolo_f4(x3)
        x4 = self.to_yolo_f5(x4)

        return x4, x3, x2

    def retina_extract(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        print(x2.shape)
        print(x3.shape)
        print(x4.shape)

        return x4, x3, x2


def resnet18(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-18 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pre_trained_path['resnet18']), strict=False)
    return model


def resnet34(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-34 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pre_trained_path['resnet34']), strict=False)
    return model


def resnet50(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pre_trained_path['resnet50']), strict=False)
    return model


def resnet101(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-101 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pre_trained_path['resnet101']), strict=False)
    return model


def resnet152(num_classes, pretrained=False, **kwargs):
    """Constructs a ResNet-152 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(num_classes, Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        model.load_state_dict(torch.load(pre_trained_path['resnet152']), strict=False)
    return model
