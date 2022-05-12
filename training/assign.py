# this .py is for the assignment methods
import torch

from .iou import IOU
from .config import cfg
from models.anchor import generateAnchors
from itertools import product


class AnchAssign():
    def __init__(self, anchors=generateAnchors(), _cfg = cfg):
        if isinstance(_cfg, str):
            from .config import Config
            _cfg = Config(_cfg)
        self.cfg = _cfg
        self.anchs = anchors
        self.anchs_len = anchors.shape[1]
        self.assignType = _cfg.assignType
        self.iou = IOU(ioutype=_cfg.iouType)
        self.threshold_iou = _cfg.threshold
    def assign(self,gt):
        '''
        using batch_sized data input
        :param gt:
        :return:the same sture of self.anchs, but filled
                with bool value
        '''
        if self.assignType == "default":
            return self._retinaAssign(gt)
        elif self.assignType == "ATSS":
            return self._ATSS(gt)
        else:
            raise NotImplementedError("Unknown assignType")

    def _retinaAssign(self,gt):
        singleAch = self.anchs[0,:]
        if torch.cuda.is_available():
            singleAch = torch.from_numpy(singleAch).double().cuda()
        else:
            singleAch = torch.from_numpy(singleAch).double()

        output_size = self.anchs.shape[:2]
        assign_result = torch.zeros(tuple(output_size))
        for ib in range(self.cfg.batch_size):
            imgAnn = gt[ib]
            lenth_gt = len(imgAnn)
            if torch.cuda.is_available():
                imgAnn = torch.from_numpy(imgAnn).double().cuda()
            else:
                imgAnn = torch.from_numpy(imgAnn).double()

            pair_ann = torch.repeat_interleave(singleAch, lenth_gt,dim=0)
            pair_gt = torch.repeat_interleave(imgAnn.unsqueeze(0), self.anchs_len, dim=0)
            pair_gt = torch.flatten(pair_gt,start_dim=0, end_dim=1)

            iou_matrix = self.iou(pair_ann,pair_gt)
            iou_matrix = iou_matrix.reshape((self.anchs_len,lenth_gt))
            look = iou_matrix.cpu().numpy()
            iou_max_value, iou_max_idx = torch.max(iou_matrix, dim=1)
            iou_max_value = torch.where(iou_max_value >= 0.5, iou_max_value, iou_max_idx.double() + 2.0)
            iou_max_value = torch.where(iou_max_value < 0.4, iou_max_value, 1.0)
            iou_max_value = torch.where(iou_max_value < 0.5, iou_max_value, 0.).int()
            assign_result[ib] = iou_max_value-1

        return assign_result

    def _ATSS(self,gt):
        pass
