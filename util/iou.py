#this .py is for Iou related algorithms

import torch
import torch.nn as nn

class Iouloss(nn.Module):
    def __init__(self, ioutype="iou"):
        super(Iouloss, self).__init__()
        self.ioutype = ioutype

    def forward(self,dt,gt):
        dt_x1 = dt[:, 0]
        dt_y1 = dt[:, 1]
        dt_x2 = dt[:, 2]
        dt_y2 = dt[:, 3]

        gt_x1 = gt[:, 0]
        gt_y1 = gt[:, 1]
        gt_x2 = gt[:, 2]
        gt_y2 = gt[:, 3]

        x_min = torch.max(dt_x1, gt_x1)
        x_max = torch.min(dt_x2, gt_x2)
        y_min = torch.max(dt_y1, gt_y1)
        y_max = torch.max(dt_y2, gt_y2)

        w_int = torch.clamp(x_max - x_min,0)
        h_int = torch.clamp(y_max - y_min,0)

        join = w_int*h_int
        union = (dt_x2-dt_x1)*(dt_y2-dt_y1)+(gt_x2-gt_x1)*(gt_y2-gt_y1)-join

        return join/union

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
        xlen = torch.clamp(xmax-xmin,0)
        ylen = torch.clamp(ymax-ymin,0)
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
