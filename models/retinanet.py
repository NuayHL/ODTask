import torch
import torch.nn as nn
import numpy as np
from models.resnet import resnet101, resnet50, resnet18, resnet34
from models.fpn import FPN
from models.common import BasicBlock, Bottleneck
from models.anchor import generateAnchors
from models.nms import NMS
from models.initialize import weight_init
from training.config import cfg
from training.loss import FocalLoss_yoloInput, FocalLoss_splitInput
from torch.distributed import get_rank, is_initialized
from data.dataset import load_single_inferencing_img
from training.eval import Results

class RetinaNet(nn.Module):
    def __init__(self, numofclass=1, loss=FocalLoss_splitInput, anchors = generateAnchors(singleBatch=True),
                 nms=NMS(), config=cfg):
        super(RetinaNet, self).__init__()
        self.config = config
        self.core = Retina_core(numofclass, config=config)
        # initialize weight
        self.core.apply(weight_init)
        self.anchors_per_grid = len(self.config.anchorRatio) * len(self.config.anchorScales)

        if is_initialized():
            print("Using MultiGPU: Process",get_rank())
            self.device = get_rank()
        else:
            print("Using SingleGPU")
            self.device = config.pre_device

        self.loss = loss(device=self.device, use_focal=False, use_ignore=True)
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        self._pre_anchor(anchors)
        self.softmax = nn.Softmax(dim=1)
        self.nms = nms

    def forward(self, input):
        if self.training:
            return self._training(input)
        else:
            return self._inferencing(input)

    def _training(self, samples):
        imgs, annos = samples["imgs"], samples["anns"]
        cls_dt, reg_dt = self.core(imgs)
        return self.loss(cls_dt, reg_dt, annos)

    def _inferencing(self, imgs):
        imgs = load_single_inferencing_img(imgs, device=self.device)
        cls_dt, reg_dt = self.core(imgs)
        cls_dt, reg_dt = cls_dt.detach(), reg_dt.detach()
        anchors = torch.tile(self.anchors, (reg_dt.shape[0], 1, 1))
        reg_dt[:, 2:4, :] = anchors[:, 2:, :] * reg_dt[:, 2:4, :]
        reg_dt[:, :2, :] = anchors[:, :2, :] + reg_dt[:, :2, :] * anchors[:, :2, :]
        result_list = []
        for ib in range(cls_dt.shape[0]):
            # delete background
            max_value, max_index = torch.max(cls_dt[ib], dim=0)
            posi_ib_idx = torch.ge(max_value, self.config.background_threshold)
            reg_dt_ib = reg_dt[ib, :, posi_ib_idx].t()
            cls_dt_ib_score = torch.unsqueeze(max_value[posi_ib_idx],dim=1)
            cls_dt_ib_cls = torch.unsqueeze(max_index[posi_ib_idx],dim=1)
            if reg_dt_ib.shape[0] == 0:
                real_result = None
            else:
                self._xywh_to_x1y1x2y2(reg_dt_ib)
                result_ib = torch.cat([reg_dt_ib, cls_dt_ib_score, cls_dt_ib_cls], dim=1)
                fin_list = self.nms(result_ib, self.config.nms_threshold)
                real_result = Results(result_ib[fin_list].detach().cpu())
            result_list.append(real_result)
        return result_list

    def _xywh_to_x1y1x2y2(self, input):
        input[:, 0] = input[:, 0] - 0.5 * input[:, 2]
        input[:, 1] = input[:, 1] - 0.5 * input[:, 3]
        input[:, 2] = input[:, 0] + input[:, 2]
        input[:, 3] = input[:, 1] + input[:, 3]
        return input

    def _pre_anchor(self, anchors):
        if torch.cuda.is_available():
            anchors = anchors.to(self.device)
        anchors[:, 2] = anchors[:, 2] - anchors[:, 0]
        anchors[:, 3] = anchors[:, 3] - anchors[:, 1]
        anchors[:, 0] = anchors[:, 0] + 0.5 * anchors[:, 2]
        anchors[:, 1] = anchors[:, 1] + 0.5 * anchors[:, 3]
        self.anchors = anchors.t()

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
        #out = self.output_act(out)
        out = torch.flatten(out, start_dim=2)
        out_split = torch.split(out, int(out.shape[1]/self.num_anchors), dim=1)
        out = torch.cat(out_split, dim=2)
        return out

class Retina_core(nn.Module):
    """
    :return (cls_dt, reg_dt)
        cls_dt: B x numofclasss x anchors
        reg_dt: B x 4 x anchors
    """
    def __init__(self, numofclasses=1, backbone=resnet101, neck_channels=256, config=cfg):
        super(Retina_core, self).__init__()
        self.anchors = len(config.anchorRatio) * len(config.anchorScales)
        self.numofclasses = numofclasses
        self.backbone = backbone(numofclasses, yolo_use=False)
        if isinstance(self.backbone.layer1[-2], Bottleneck):
            self.fpn = FPN([512, 1024, 2048], feature_size=neck_channels)
        elif isinstance(self.backbone.layer1[-2], BasicBlock):
            self.fpn = FPN([128, 256, 512], feature_size=neck_channels)
        else:
            raise NotImplementedError("unsupport blocks")

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
    model = Retina_core(backbone=resnet18)
    model = model.cuda()
    input = torch.rand((4, 3, 800, 1024))
    input = input.cuda()
    cls, reg = model(input)
    print(cls.shape, reg.shape)

