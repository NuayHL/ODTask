import torch.nn as nn

class Conv(nn.Module):
    def __init__(self):
        self.conv = nn.Conv2d(3, 16, 3, 1, 1)
        self.bn = nn.BatchNorm2d()
        self.act = nn.LeakyReLU()

    def forward(self,x):
        return self.act(self.bn(self.conv(x)))
