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
    def __init__(self, anchors=generateAnchors(singleBatch=True), config = cfg, using_ignored_input=False,
                 device=None):
        if isinstance(config, str):
            from .config import Config
            config = Config(config)
        self.cfg = config
        self.anchs = torch.from_numpy(anchors).double()
        self.anchs_len = anchors.shape[0]
        self.assignType = config.assignType.lower()
        self.iou = IOU(ioutype=config.assignIouType, gt_type='x1y1wh')
        self.threshold_iou = config.assign_threshold
        self.using_ignored_input = using_ignored_input

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
            if self.using_ignored_input:
                return self._retinaAssign_using_ignored(gt)
            else:
                print("Warning: this assign method can not handle ignored anns")
                return self._retinaAssign(gt)
        elif self.assignType == "atss":
            return self._ATSS(gt)
        elif self.assignType == "freeanchor":
            return self._freeanchor(gt)
        else:
            raise NotImplementedError("Unknown assignType: %s"%self.assignType)

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

    def _retinaAssign_using_ignored(self,gt):
        ''':return: assign result, real gt(exclude ignored ones), all already to tensor'''
        # initialize assign result
        output_size = (len(gt),self.anchs.shape[0])
        assign_result = torch.zeros(output_size)
        if torch.cuda.is_available():
            assign_result = assign_result.to(self.device)

        # prepare return real gt
        real_gt = []

        for ib in range(len(gt)):
            gt_i = torch.from_numpy(gt[ib]).double()
            if torch.cuda.is_available():
                gt_i = gt_i.to(self.device)

            ignored = torch.eq(gt_i[:, 4].int(), 0)
            real_gt.append(gt_i[~ignored])
            imgAnn = gt_i[:, :4]
            ignoredAnn = imgAnn[ignored]
            imgAnn = imgAnn[~ignored]

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
            iou_max_value = iou_max_value.int()

            # Dealing with ignored area
            if ignoredAnn.shape[0] != 0:
                false_sample_idx = torch.eq(iou_max_value, 1)
                ignore_iou_matrix = self.iou(self.anchs[false_sample_idx], ignoredAnn)
                false_sample = iou_max_value[false_sample_idx]
                ignored_max_iou_value, _ = torch.max(ignore_iou_matrix, dim=1)
                # set the iou threshold as 0.5
                ignored_anchor_idx = torch.ge(ignored_max_iou_value, 0.5)
                false_sample[ignored_anchor_idx] = 0
                iou_max_value[false_sample_idx] = false_sample

            assign_result[ib] = iou_max_value-1

        return assign_result, real_gt

    def _ATSS(self,gt):
        pass

    def _freeanchor(self,gt):
        pass
