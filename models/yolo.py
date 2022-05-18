from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import interpolate

from .backbone import Darknet53
from .common import conv_batch
from training.loss import Defaultloss
from training.config import cfg
from models.nms import NMS

anchors_per_grid = len(cfg.anchorRatio) * len(cfg.anchorScales)

class YOLOv3(nn.Module):
    def __init__(self, numofclasses=2, ioutype="iou", loss=Defaultloss(), nms=NMS(), istrainig=False):
        super(YOLOv3, self).__init__()
        self.numofclasses = numofclasses
        self.core = Yolov3_core(numofclasses)
        self.loss = loss
        self.nms = nms
        self.istraining = istrainig
    def forward(self,x):
        '''
        when training, x: tuple(img(Tensor),annos)
        when inferencing, x:
        '''
        if not self.istraining:
            return self.inference(x)
        else:
            input, anno = x
            result = self.core(input)
            return self.cal_loss(input,anno)

    def inference(self,x):
        inputtensor = self.core(x)
        return self.nms(inputtensor)

    def cal_loss(self,result,anno):

        return self.loss(result,anno)

    def _result_parse(self,triple):
        '''
        flatten the results according to the format of anchors
        '''
        base_len = int(triple[0].shape[1]/anchors_per_grid)
        out = torch.zeros((triple[0].shape[0],0, int(5 + self.numofclasses)))
        for fp in triple:
            fp = torch.flatten(fp,start_dim=2)
            split = torch.split(fp,int(fp.shape[1]/anchors_per_grid),dim=1)
            out = torch.stack(split,dim=-1)





class Yolov3_core(nn.Module):
    def __init__(self, numofclasses=2, backbone=Darknet53):
        super(Yolov3_core, self).__init__()
        self.backbone = backbone(numofclasses)
        self.extractor = backbone.yolo_extract
        self.yolodetector1 = self.yolo_block(512)
        self.to_featuremap1 = nn.Sequential(
            conv_batch(512, 1024),
            conv_batch(1024, (1 + 4 + numofclasses) * anchors_per_grid, kernel_size=1, padding=0))
        self.yolodetector2 = self.yolo_block(256,768)
        self.to_featuremap2 = nn.Sequential(
            conv_batch(256, 512),
            conv_batch(512, (1 + 4 + numofclasses) * anchors_per_grid, kernel_size=1, padding=0))
        self.yolodetector3 = self.yolo_block(128,384)
        self.to_featuremap3 = nn.Sequential(
            conv_batch(128, 256),
            conv_batch(256, (1 + 4 + numofclasses) * anchors_per_grid, kernel_size=1, padding=0))
        self.conv1to2 = conv_batch(512, 256, kernel_size=1, padding=0)
        self.conv2to3 = conv_batch(256, 128, kernel_size=1, padding=0)

    def yolo_block(self, channel, ic: Optional[int]=None):
        if not ic:
            ic = channel*2
        return nn.Sequential(
            conv_batch(ic, channel, kernel_size=1, padding=0),
            conv_batch(channel, channel * 2),
            conv_batch(channel * 2, channel, kernel_size=1, padding=0),
            conv_batch(channel, channel * 2),
            conv_batch(channel * 2, channel, kernel_size=1, padding=0))

    def forward(self,x):
        '''
        :param x: BxCxWxH tensor
        :return:
        '''
        f1, f2, f3 = self.extractor(self.backbone, x)
        f1 = self.yolodetector1(f1)  #large grid
        f1_up = interpolate(self.conv1to2(f1), size=(f2.shape[2],f2.shape[3]))
        f2 = torch.cat((f2,f1_up),1)
        f2 = self.yolodetector2(f2)  # middle grid
        f2_up = interpolate(self.conv2to3(f2), size=(f3.shape[2],f3.shape[3]))
        f3 = torch.cat((f3,f2_up),1)
        f3 = self.yolodetector3(f3)  # small grid

        f1 = self.to_featuremap1(f1) # W/32 1024
        f2 = self.to_featuremap2(f2) # W/16 512
        f3 = self.to_featuremap3(f3) # W/8  256

        return f3, f2, f1

