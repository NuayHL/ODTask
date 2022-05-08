# this .py is for the assignment methods
from .iou import IOU
from .config import cfg

class Assign():
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
        elif self.assignType == "FCOS":
            return self._focsAssign(gt)
        else:
            raise NotImplementedError("Unknown assignType")

    def _retinaAssign(self,gt):
        self.lenGt = len(gt)
        
        

    def _fcosAssign(self,gt):
        pass
