from typing import Optional

import torch
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.ops import batched_nms
import numpy as np

from .backbone import Darknet53
from .common import conv_batch
from training.loss import Defaultloss
from training.config import cfg
from models.nms import NMS
from models.anchor import generateAnchors
from data.trandata import load_single_inferencing_img
from data.eval import Results

from torch.distributed import get_rank, is_initialized


anchors_per_grid = len(cfg.anchorRatio) * len(cfg.anchorScales)

class YOLOv3(nn.Module):
    def __init__(self, numofclasses=2, ioutype="iou", loss=Defaultloss(), nms=NMS(), anchors = generateAnchors(singleBatch=True),istrainig=False):
        super(YOLOv3, self).__init__()
        self.numofclasses = numofclasses
        self.core = Yolov3_core(numofclasses)
        self.loss = loss
        self.nms = nms
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        self._pre_anchor(anchors)
        self.istraining = istrainig

    def _pre_anchor(self, anchors):
        if torch.cuda.is_available():
            anchors =  anchors.to(cfg.pre_device)
        anchors[:, 2] = anchors[:, 2] - anchors[:, 0]
        anchors[:, 3] = anchors[:, 3] - anchors[:, 1]
        anchors[:, 0] = anchors[:, 0] + 0.5 * anchors[:, 2]
        anchors[:, 1] = anchors[:, 1] + 0.5 * anchors[:, 3]
        self.anchors = anchors.t()

    def forward(self,x):
        '''
        when training, x: tuple(img(Tensor),annos)
        when inferencing, x: imgs(Tensor)
        '''
        if not self.istraining:
            return self.inference(x)
        else:
            input, anno = x["imgs"], x["anns"]
            result = self.core(input)
            return self.cal_loss(result,anno)

    def inference(self,x):
        x = load_single_inferencing_img(x)
        if torch.cuda.is_available():
            x = x.to(cfg.pre_device)
        result = self.core(x)
        dt = self._result_parse(result).detach()

        anchors = torch.tile(self.anchors, (dt.shape[0], 1, 1))

        # restore the predicting bboxes via pre-defined anchors
        dt[:,2:4,:] = anchors[:,2:, :] * torch.exp(dt[:,2:4,:])
        dt[:,:2,:] = anchors[:, 2, :] + dt[:,:2,:] * anchors[:,2:, :]
        #dt[:, 4:, :] = -dt[:, 4:, :]
        dt = torch.clamp(dt,min = 0)
        dt[:,4:,:] = torch.clamp(dt[:,4:,:],max = 1.)

        result_list = []
        posi_idx = torch.ge(dt[:, 4, :], cfg.background_threshold)
        for ib in range(dt.shape[0]):
            # delete background
            dt_ib = dt[ib,:,posi_idx[ib]]

            # delete low score
            max_value, max_index = torch.max(dt_ib[5:,:], dim = 0)
            has_object_idx = torch.ge(max_value, cfg.class_threshold)
            sum1 = has_object_idx.sum()
            max_value = max_value[has_object_idx]
            max_index = max_index[has_object_idx]

            dt_ib = dt_ib[:4, has_object_idx].t()

            if dt_ib.shape[0] == 0:
                real_result = None
            else:
                dt_ib = self._xywh_to_x1y1x2y2(dt_ib)
                fin_list = batched_nms(dt_ib, max_value, max_index, cfg.nms_threshold)
                real_result = Results(dt_ib[fin_list], max_index[fin_list], max_value[fin_list])
            result_list.append(real_result)

        return result_list

    def cal_loss(self,result,anno):
        result = self._result_parse(result)
        return self.loss(result,anno)

    def _result_parse(self,triple):
        '''
        flatten the results according to the format of anchors
        '''
        out = torch.zeros((triple[0].shape[0], int(5 + self.numofclasses),0))
        if torch.cuda.is_available():
            # if using DDP
            if is_initialized():
                out = out.to(get_rank())
            else:
                out = out.to(cfg.pre_device)
        for fp in triple:
            fp = torch.flatten(fp,start_dim=2)
            split = torch.split(fp,int(fp.shape[1]/anchors_per_grid),dim=1)
            split = torch.cat(split,dim=2)
            out = torch.cat((out,split),dim=2)
        return out

    def _xywh_to_x1y1x2y2(self,input):
        input[:, 0] = input[:,0] - 0.5 * input[:,2]
        input[:, 1] = input[:,1] - 0.5 * input[:,3]
        input[:, 2] = input[:,0] + input[:,2]
        input[:, 3] = input[:,1] + input[:,3]

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

