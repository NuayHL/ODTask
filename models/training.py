import torch

ZEROTENSOR = torch.Tensor(0).cuda()

class Iou():
    def __init__(self, iou="iou", ip1type="default", ip2type="default"):
        assert ip1type and ip2type in ["default", "crowdhuman", "diagonal"], "unknown format"
        self.ip1 = ip1type
        self.ip2 = ip2type
        self.type = iou

    def _tranInput(self, ):
        pass

    # need to compute for loss backward, how?
    def iou(self, bbox1, bbox2):
        xmin = max(bbox1[0],bbox2[0])
        xmax = min(bbox1[2],bbox2[2])
        ymin = max(bbox1[1],bbox2[1])
        ymax = min(bbox1[3],bbox2[3])
        xlen = max(ZEROTENSOR,xmax-xmin)
        ylen = max(ZEROTENSOR,ymax-ymin)
        join = xlen*ylen
        to = (bbox1[2]-bbox1[0])*(bbox1[3]-bbox1[1])+(bbox2[2]-bbox2[0])*(bbox2[3]-bbox2[1])-join
        return join/to

    def make(self):
        return self.iou

class Assign():
    def __init__(self):
        pass
    def _retinaAssign(self,dt,gt):
        pass
