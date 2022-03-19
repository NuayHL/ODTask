import torchvision.models
import torch.nn as nn

from common import conv_batch
from common import DarkResidualBlock
from common import make_layers

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

    def forward(self,x):
        x = self.conv1(x)
        x = self.residual_block1(self.conv2(x))
        x = self.residual_block2(self.conv3(x))
        x = self.residual_block3(self.conv4(x))
        x = self.residual_block4(self.conv5(x))
        x = self.residual_block5(self.conv6(x))
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def extract(self):
        '''
        feed data in appropriate format to the detector
        '''
        pass

