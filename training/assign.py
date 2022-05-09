# this .py is for the assignment methods
from .iou import IOU
from .config import cfg

class AnchAssign():
    def __init__(self, _cfg = cfg):
        if isinstance(_cfg, str):
            from .config import Config
            _cfg = Config(_cfg)
        self.anchs = _cfg.anchs
        self.lenAnchs = len(anchs)
        self.assignType = _cfg.assignType
        self.iou = IOU(ioutype=ioutype)
        self.threshold_iou = 0.5
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
        self.lenGt = len(gt)
        
        

    def _ATSS(self,gt):
        pass
