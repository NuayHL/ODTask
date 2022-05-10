# this .py is for the assignment methods
import torch

from .iou import IOU
from .config import cfg
from models.anchor import generateAnchors

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
        output_size = self.anchs.shape
        output_size[-1] = 1
        assign = torch.zeros(tuple(output_size))
        for ib in range(self.cfg.batch_size):
            gti = gt[ib]

        

    def _ATSS(self,gt):
        pass
