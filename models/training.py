import torch.nn.functional as F

class Iou():
    def __init__(self,iou="iou",ip1="default",ip2="default"):
        assert ip1 and ip2 in ["default","crowdhuman","diagonal"],"unknown format"
        self.ip1 = ip1
        self.ip2 = ip2
        self.iou = iou

    def _tranInput(self):
        pass

def iou(a, b, type="default"):
    assert len(a)==4 and len(b)==4,"bbox format unknown"
    if type=="crowdhuman":
        a[2], a[3] = a[0]+a[2], a[1]+a[3]


class Assign():
    def __init__(self):
        pass
    def _retinaAssign(self,dt,gt):
        pass

def yolo_loss(model_output, ground_truth):
    pass

def yolotraining(yolomodel, training_data, optimizer, loss=yolo_loss()):
    pass