import torch.nn as nn
import math

from .common import conv_batch
from .common import DarkResidualBlock
from .common import make_layers

class Darknet53(nn.Module):
    def __init__(self, numofclasses, res_block=DarkResidualBlock):
        super(Darknet53, self).__init__()
        self.numofclasses = numofclasses

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = make_layers(1, res_block, 64)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = make_layers(2, res_block, 128)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = make_layers(8, res_block, 256)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = make_layers(8, res_block, 512)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = make_layers(4, res_block, 1024)
        self.avg_pooling = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(1024, self.numofclasses)

        # initialize parameters
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self,x):
        x = self.conv1(x)
        # w 32
        x = self.residual_block1(self.conv2(x))
        # w/2 64
        x = self.residual_block2(self.conv3(x))
        # w/4 128
        x = self.residual_block3(self.conv4(x))
        # w/8 256
        x = self.residual_block4(self.conv5(x))
        # w/16 512
        x = self.residual_block5(self.conv6(x))
        # w/32 1024
        x = self.avg_pooling(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def yolo_extract(self,x):
        '''
        w = w/8
        return w*w*256 (w/2)*(w/2)*512 (w/4)*(w/4)*1024
        '''
        x = self.conv1(x)
        x = self.residual_block1(self.conv2(x))
        x = self.residual_block2(self.conv3(x))
        f1 = self.residual_block3(self.conv4(x))  #256
        f2 = self.residual_block4(self.conv5(f1)) #512
        f3 = self.residual_block5(self.conv6(f2)) #1024
        return f3, f2, f1

