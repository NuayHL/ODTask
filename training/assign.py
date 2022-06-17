# this .py is for the assignment methods
import torch

from .iou import IOU
from .config import cfg
from models.anchor import generateAnchors

'''
WARNING:
All the assign class must have "config" as initialized parameter
'''

class AnchAssign():
    def __init__(self, anchors=generateAnchors(singleBatch=True), config = cfg, device=None):
        if isinstance(config, str):
            from .config import Config
            config = Config(config)
        self.cfg = config
        self.anchs = torch.from_numpy(anchors).double()
        self.anchs_len = anchors.shape[0]
        self.assignType = config.assignType
        self.iou = IOU(ioutype=config.assignIouType, gt_type='x1y1wh')
        self.threshold_iou = config.assign_threshold

        if device is None:
            self.device = config.pre_device
        else:
            self.device = device

        if torch.cuda.is_available():
            self.anchs = self.anchs.to(self.device)

    def assign(self,gt):
        '''
        using batch_sized data input
        :param gt:aka:"anns":List lenth B, each with np.float32 ann}
        :return:the same sture of self.anchs, but filled
                with value indicates the assignment of the anchor
        '''
        if self.assignType == "default":
            return self._retinaAssign(gt)
        elif self.assignType == "ATSS":
            return self._ATSS(gt)
        else:
            raise NotImplementedError("Unknown assignType")

    def _retinaAssign(self,gt):
        output_size = (len(gt),self.anchs.shape[0])
        assign_result = torch.zeros(output_size)
        if torch.cuda.is_available():
            assign_result = assign_result.to(self.device)
        for ib in range(len(gt)):
            imgAnn = gt[ib][:,:4]
            imgAnn = torch.from_numpy(imgAnn).double()
            if torch.cuda.is_available():
                imgAnn = imgAnn.to(self.device)

            iou_matrix = self.iou(self.anchs, imgAnn)
            iou_max_value, iou_max_idx = torch.max(iou_matrix, dim=1)
            iou_max_value_anns, iou_max_idx_anns = torch.max(iou_matrix, dim=0)
            # negative: 0
            # ignore: -1
            # positive: index+1
            iou_max_value = torch.where(iou_max_value >= 0.5, iou_max_idx.double() + 2.0,iou_max_value)
            iou_max_value = torch.where(iou_max_value < 0.4, 1.0, iou_max_value)
            iou_max_value = torch.where(iou_max_value < 0.5, 0., iou_max_value)

            # Assign at least one anchor to the gt
            iou_max_value[iou_max_idx_anns] = torch.arange(imgAnn.shape[0]).double().to(self.device) + 2
            assign_result[ib] = iou_max_value-1

        return assign_result

    def _ATSS(self,gt):
        pass
