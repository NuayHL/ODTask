# this .py is for the assignment methods
from .iou import IOU

class Assign():
    def __init__(self,anchs,assignType="default",ioutype="iou"):
        self.anchs = anchs
        self.lenAnchs = len(anchs)
        self.assignType = assignType
        self.iou = IOU(ioutype=ioutype)
    def assign(self,gt):
        if self.assignType == "default":
            return self._retinaAssign(gt)
        elif self.assignType == "FCOS":
            return self._focsAssign(gt)
        else:
            raise NotImplementedError("Unknown assignType")

    def _retinaAssign(self,gt):
        lenGt = len(gt)
        

    def _fcosAssign(self,gt):
        pass
