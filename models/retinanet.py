import torch
import torch.nn as nn
from models.resnet import resnet101
from models.fpn import FPN
from training.config import cfg

class RetinaNet(nn.Module):
    def __init__(self, numofclass=1, config=cfg):
        super(RetinaNet, self).__init__()
        self.config = config
        self.core = Retina_core(numofclass, config=config)

    def forward(self, input):
        if self.training:
            return self._training(input)
        else:
            return self._inferencing(input)

    def _training(self, samples):
        pass

    def _inferencing(self, imgs):
        pass

class RegressionModule(nn.Module):
    def __init__(self, in_channels, num_anchors=4, out_channels=256):
        super(RegressionModule, self).__init__()
        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(out_channels, num_anchors * 4, kernel_size=3, padding=1)
    def forward(self, x):
        out = self.conv1(x)
        out = self.act1(out)

        out = self.conv2(out)
        out = self.act2(out)

        out = self.conv3(out)
        out = self.act3(out)

        out = self.conv4(out)
        out = self.act4(out)

        out = self.output(out)
        out = torch.flatten(out, start_dim=2)
        out_split = torch.split(out, int(out.shape[1]/self.num_anchors), dim=1)
        out = torch.cat(out_split, dim=2)
        return out

class ClassificationModule(nn.Module):
    def __init__(self, numofclass=1, num_anchors=4):
        super(ClassificationModule, self).__init__()
    def forward(self):
        pass

class Retina_core(nn.Module):
    def __init__(self, numofclasses=1, backbone=resnet101, config=cfg):
        self.backbone = backbone(numofclasses, yolo_use=False)
        self.fpn = FPN([512, 1024, 2048])

    def forward(self, x):
        p5, p4, p3 = self.backbone.retina_extract(x)
        p3_7 = self.fpn((p3, p4, p5))

if __name__ == "__main__":
    model = RegressionModule(256)
    input = torch.rand((4, 256, 32, 32))
    out = model(input)
    print(out.shape)

