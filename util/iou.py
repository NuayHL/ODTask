#this .py is for Iou related algorithms

import torch

ZEROTENSOR = torch.Tensor([0]).cuda()

class Iou():
    def __init__(self, ioutype="iou", ip1type="default", ip2type="default"):
        assert ip1type and ip2type in ["default", "crowdhuman", "diagonal"], "Unknown bbox format"
        assert ioutype in ["iou","giou"],"Unknow iou type"
        self.ip1 = ip1type
        self.ip2 = ip2type
        self.ioutype = ioutype

    def _tranInput(self, ):
        pass

    # need to compute for loss backward, how?
    def iou(self, bbox1, bbox2):
        xmin = torch.max(bbox1[0:1],bbox2[0:1])
        xmax = torch.min(bbox1[2:3],bbox2[2:3])
        ymin = torch.max(bbox1[1:2],bbox2[1:2])
        ymax = torch.min(bbox1[3:4],bbox2[3:4])
        xlen = torch.max(ZEROTENSOR,xmax-xmin)
        ylen = torch.max(ZEROTENSOR,ymax-ymin)
        join = xlen*ylen
        to = (bbox1[2:3]-bbox1[0:1])*(bbox1[3:4]-bbox1[1:2])+(bbox2[2:3]-bbox2[0:1])*(bbox2[3:4]-bbox2[1:2])-join
        return join/to

    def giou(self, bbox1, bbox2):
        pass

    def make(self):
        if self.ioutype == "iou":
            return self.iou
        elif self.ioutype == "giou":
            return self.giou

class Assign():
    def __init__(self):
        pass
    def _retinaAssign(self,dt,gt):
        pass
