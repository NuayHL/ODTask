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
    def __init__(self, in_channels, numofclasses=1, num_anchors=4, out_channels=256):
        super(ClassificationModule, self).__init__()

        self.num_anchors = num_anchors

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.act1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()

        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()

        self.output = nn.Conv2d(out_channels, num_anchors * numofclasses, kernel_size=3, padding=1)
        self.output_act = nn.Sigmoid()

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
        out = self.output_act(out)
        out = torch.flatten(out, start_dim=2)
        out_split = torch.split(out, int(out.shape[1]/self.num_anchors), dim=1)
        out = torch.cat(out_split, dim=2)
        return out

class Retina_core(nn.Module):
    def __init__(self, numofclasses=1, backbone=resnet101, neck_channels=256, config=cfg):
        super(Retina_core, self).__init__()
        self.anchors = len(config.anchorRatio) * len(config.anchorScales)
        self.numofclasses = numofclasses
        self.backbone = backbone(numofclasses, yolo_use=False)
        self.fpn = FPN([512, 1024, 2048], feature_size=neck_channels)
        self.regression = RegressionModule(neck_channels, num_anchors=self.anchors)
        self.classification = ClassificationModule(neck_channels, numofclasses=self.numofclasses,
                                                   num_anchors = self.anchors)

    def forward(self, x):
        p5, p4, p3 = self.backbone.retina_extract(x)
        p3_7 = self.fpn((p3, p4, p5))
        cls_dt = [self.classification(fmp) for fmp in p3_7]
        reg_dt = [self.regression(fmp) for fmp in p3_7]
        cls_dt = torch.cat(cls_dt, dim=2)
        reg_dt = torch.cat(reg_dt, dim=2)
        return cls_dt, reg_dt

if __name__ == "__main__":
    model = Retina_core()
    model = model.cuda()
    input = torch.rand((4, 3, 512, 512))
    input = input.cuda()
    cls, reg = model(input)
    print(cls.shape, reg.shape)

