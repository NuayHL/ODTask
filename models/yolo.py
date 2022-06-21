from typing import Optional

import torch
import math
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.ops import batched_nms
import numpy as np

from .darknet53 import Darknet53
from .common import conv_batch
from training.loss import Defaultloss
from training.config import cfg
from models.nms import NMS
from models.anchor import generateAnchors
from data.dataset import load_single_inferencing_img
from training.eval import Results

from torch.distributed import get_rank, is_initialized

class YOLOv3(nn.Module):
    def __init__(self, numofclasses=2, loss=Defaultloss, nms=NMS(),
                 anchors = generateAnchors(singleBatch=True), istrainig=False, backbone=None,
                 config=cfg, **kwargs):
        super(YOLOv3, self).__init__()
        if backbone == None:
            backbone = Darknet53
        self.numofclasses = numofclasses
        self.config = config
        self.anchors_per_grid = len(self.config.anchorRatio) * len(self.config.anchorScales)
        self.core = Yolov3_core(numofclasses, backbone=backbone, anchors_per_grid=self.anchors_per_grid, **kwargs)
        self.nms = nms
        if is_initialized():
            print("MultiGpuTraining: Process",get_rank())
            self.device = get_rank()
        else:
            print("SingleGpuTraining..")
            self.device = config.pre_device
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        self.loss = loss(device=self.device, use_focal=False)
        self._pre_anchor(anchors)
        self.istraining = istrainig
        self.softmax = nn.Softmax(dim=1)

    def _pre_anchor(self, anchors):
        if torch.cuda.is_available():
            anchors = anchors.to(self.device)
        anchors[:, 2] = anchors[:, 2] - anchors[:, 0]
        anchors[:, 3] = anchors[:, 3] - anchors[:, 1]
        anchors[:, 0] = anchors[:, 0] + 0.5 * anchors[:, 2]
        anchors[:, 1] = anchors[:, 1] + 0.5 * anchors[:, 3]
        self.anchors = anchors.t()

    def forward(self, x):
        '''
        when training, x: tuple(img(Tensor),annos)
        when inferencing, x: imgs(Tensor)
        '''
        if not self.istraining:
            return self.inference(x)
        else:
            input, anno = x["imgs"], x["anns"]
            result = self.core(input)
            return self.cal_loss(result, anno)

    def inference(self, x):
        x = load_single_inferencing_img(x)
        if torch.cuda.is_available():
            x = x.to(self.device)
        result = self.core(x)
        dt = self._result_parse(result).detach()

        anchors = torch.tile(self.anchors, (dt.shape[0], 1, 1))

        # restore the predicting bboxes via pre-defined anchors
        dt[:, 2:4, :] = anchors[:, 2:, :] * torch.exp(dt[:, 2:4, :])
        dt[:, :2, :] = anchors[:, :2, :] + dt[:, :2, :] * anchors[:, 2:, :]
        dt[:, 4, :] = torch.clamp(dt[:, 4, :], min=0., max=1.)
        dt[:, 5:, :] = self.softmax(dt[:, 5:, :])

        result_list = []
        posi_idx = torch.ge(dt[:, 4, :], self.config.background_threshold)
        for ib in range(dt.shape[0]):
            # delete background
            dt_ib = dt[ib, :, posi_idx[ib]]

            max_value, max_index = torch.max(dt_ib[5:, :], dim=0)

            # fourth value is the target_scores
            dt_ib = dt_ib[:5, :].t()
            result_ib = torch.cat([dt_ib, torch.unsqueeze(max_index,1)],dim=1)

            if result_ib.shape[0] == 0:
                real_result = None
            else:
                self._xywh_to_x1y1x2y2(result_ib[:,:4])
                fin_list = batched_nms(result_ib[:,:4], result_ib[:,4], result_ib[:,5], self.config.nms_threshold)
                real_result = Results(result_ib[fin_list].detach().cpu())
            result_list.append(real_result)

        return result_list

    def cal_loss(self, result, anno):
        result = self._result_parse(result)
        return self.loss(result, anno)

    def _result_parse(self, triple):
        '''
        flatten the results according to the format of anchors
        '''
        out = torch.zeros((triple[0].shape[0], int(5 + self.numofclasses), 0))
        if torch.cuda.is_available():
            out = out.to(self.device)
        for fp in triple:
            fp = torch.flatten(fp, start_dim=2)
            split = torch.split(fp, int(fp.shape[1] / self.anchors_per_grid), dim=1)
            split = torch.cat(split, dim=2)
            out = torch.cat((out, split), dim=2)
        return out

    def _xywh_to_x1y1x2y2(self, input):
        input[:, 0] = input[:, 0] - 0.5 * input[:, 2]
        input[:, 1] = input[:, 1] - 0.5 * input[:, 3]
        input[:, 2] = input[:, 0] + input[:, 2]
        input[:, 3] = input[:, 1] + input[:, 3]
        return input


class Yolov3_core(nn.Module):
    def __init__(self, numofclasses=2, backbone=Darknet53, anchors_per_grid=4, **kwargs):
        super(Yolov3_core, self).__init__()
        self.yolodetector1 = self.yolo_block(512)
        self.to_featuremap1 = nn.Sequential(
            conv_batch(512, 1024),
            conv_batch(1024, (1 + 4 + numofclasses) * anchors_per_grid, kernel_size=1, padding=0))
        self.yolodetector2 = self.yolo_block(256, 768)
        self.to_featuremap2 = nn.Sequential(
            conv_batch(256, 512),
            conv_batch(512, (1 + 4 + numofclasses) * anchors_per_grid, kernel_size=1, padding=0))
        self.yolodetector3 = self.yolo_block(128, 384)
        self.to_featuremap3 = nn.Sequential(
            conv_batch(128, 256),
            conv_batch(256, (1 + 4 + numofclasses) * anchors_per_grid, kernel_size=1, padding=0))
        self.conv1to2 = conv_batch(512, 256, kernel_size=1, padding=0)
        self.conv2to3 = conv_batch(256, 128, kernel_size=1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.backbone = backbone(numofclasses, **kwargs)

    def yolo_block(self, channel, ic: Optional[int] = None):
        if not ic:
            ic = channel * 2
        return nn.Sequential(
            conv_batch(ic, channel, kernel_size=1, padding=0),
            conv_batch(channel, channel * 2),
            conv_batch(channel * 2, channel, kernel_size=1, padding=0),
            conv_batch(channel, channel * 2),
            conv_batch(channel * 2, channel, kernel_size=1, padding=0))

    def forward(self, x):
        '''
        :param x: BxCxWxH tensor
        :return:
        '''
        # the extractor is required to generate 3 feature layers
        f1, f2, f3 = self.backbone.yolo_extract(x)
        f1 = self.yolodetector1(f1)  # large grid
        f1_up = interpolate(self.conv1to2(f1), size=(f2.shape[2], f2.shape[3]))
        f2 = torch.cat((f2, f1_up), 1)
        f2 = self.yolodetector2(f2)  # middle grid
        f2_up = interpolate(self.conv2to3(f2), size=(f3.shape[2], f3.shape[3]))
        f3 = torch.cat((f3, f2_up), 1)
        f3 = self.yolodetector3(f3)  # small grid

        f1 = self.to_featuremap1(f1)  # W/32 1024
        f2 = self.to_featuremap2(f2)  # W/16 512
        f3 = self.to_featuremap3(f3)  # W/8  256

        return f3, f2, f1
