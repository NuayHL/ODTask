import torch.nn as nn

'''
ic: in channels
oc: out channels
kn: kernel size
st: stride
pd: padding
'''

'''YOLO RELATED'''
def conv_batch(ic, oc, kernel_size=3, stride=1, padding=1):
    '''
    Caution: the conv layer does not contain bais.
    :return: nn.Sequential (conv, bn, activateFunction)
    the feature size doesn't change if use default par
    '''
    return nn.Sequential(
        nn.Conv2d(ic, oc, kernel_size, stride, padding, bias=False),
        nn.BatchNorm2d(oc),
        nn.LeakyReLU(inplace=True))

def make_layers(num, block, *args, **kwargs):
    '''
    :param num: the number of the blocks
    :param block: nn.Module, the type of the block
    :param args: refer to the block
    :param kwargs: refer to the block
    :return: nn.Sequential
    '''
    layers = []
    for i in range(num):
        layers.append(block(*args, **kwargs))
    return nn.Sequential(*layers)

class DarkResidualBlock(nn.Module):
    '''
    ic: in channel
    feature size remain the same
    '''
    def __init__(self, ic):
        super(DarkResidualBlock, self).__init__()
        reduced_channels = int(ic/2)
        self.conv1 = conv_batch(ic, reduced_channels, kernel_size=1, padding=0)
        self.conv2 = conv_batch(reduced_channels, ic, kernel_size=3, padding=1)

    def forward(self,x):
        residual = x
        x = self.conv2(self.conv1(x))
        return x+residual
'''YOLO RELATED END'''

'''RESNET RELATED'''
# Part from:
#     https://github.com/yhenon/pytorch-retinanet/blob/master/retinanet/utils.py

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
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


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out
'''RESNET RELATED END'''





