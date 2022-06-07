import torch
import torch.nn as nn
import numpy as np
from training.config import cfg
from training.assign import AnchAssign
from models.anchor import generateAnchors

'''
super par:
    focal loss: alpha, gamma
    cfg
'''

class Defaultloss(nn.Module):
    '''
    {"imgs":List lenth B, each with np.float32 img
     "anns":List lenth B, each with np.float32 ann}
    '''
    def __init__(self, assign_method=AnchAssign, anchors=generateAnchors(singleBatch=True), use_focal=True,
                 device=None, config = cfg):
        super(Defaultloss, self).__init__()
        if isinstance(config, str):
            from .config import Config
            _cfg = Config(config)
        self.label_assignment = assign_method(config=config, device=device)
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        self.anchs = anchors

        if device is None:
            self.device = config.pre_device
        else:
            self.device = device

        self._pre_anchor()
        self.alpha = 0.25
        self.gamma = 2.0
        self.usefocal = use_focal

    def _pre_anchor(self):
        if torch.cuda.is_available():
            self.anchs = self.anchs.to(self.device)
        self.anch_w = self.anchs[:, 2] - self.anchs[:, 0]
        self.anch_h = self.anchs[:, 3] - self.anchs[:, 1]
        self.anch_x = self.anchs[:, 0] + 0.5 * self.anch_w
        self.anch_y = self.anchs[:, 1] + 0.5 * self.anch_h

    def forward(self, dt, gt):
        '''
        dt: The input dt should be the same format as assign result
            i.e.  Tensor: Batchsize X (4+1+classes) X samples
        gt: list, each is np.array, with shape (4+1) at dim=-1
        '''
        self.classes = dt.shape[1] - 5
        self.batch_size = dt.shape[0]

        bbox_loss = []
        cls_loss = []

        assign_result = self.label_assignment.assign(gt)

        if torch.cuda.is_available():
            dt = dt.to(self.device)

        dt_class_md = torch.clamp(dt[:, 4:, :], 1e-4, 1.0 - 1e-4).clone()

        for ib in range(dt.shape[0]):

            positive_idx_cls = torch.ge(assign_result[ib], -0.1)
            # the not ignored ones
            positive_idx_box = torch.ge(assign_result[ib] - 1.0, -0.1)
            # the assigned ones

            imgAnn = gt[ib]
            imgAnn = torch.from_numpy(imgAnn).float()
            if torch.cuda.is_available():
                imgAnn = imgAnn.to(self.device)

            assign_result_box = assign_result[ib][positive_idx_box].long()-1
            assigned_anns = imgAnn[assign_result_box]

            # cls loss
            assign_result_cal = torch.clamp(assign_result[ib][positive_idx_cls], 0., 1.)
            one_hot_bed = torch.zeros((assign_result.shape[1], self.classes), dtype=torch.int64)
            if torch.cuda.is_available():
                one_hot_bed = one_hot_bed.to(self.device)

            one_hot_bed[positive_idx_box, assigned_anns[:, 4].long() - 1] = 1

            ## using 'from torch.nn.functional import one_hot'
            # one_hot_bed[positive_idx_box] = one_hot(assigned_anns[:, 4].long() - 1,
            #                                         num_classes=self.classes)

            assign_result_cal = torch.cat((torch.unsqueeze(assign_result_cal, dim=1),
                                          one_hot_bed[positive_idx_cls]), dim=1)

            dt_cls = dt_class_md[ib, :, positive_idx_cls].t()

            ## gei wo zheng wu yu le: mistake loss
            # cls_loss_ib = -dt_cls * torch.log(assign_result_cal) + \
            #               (dt_cls - 1.0) * torch.log(1.0 - assign_result_cal)
            cls_loss_ib = - assign_result_cal * torch.log(dt_cls) + \
                           (assign_result_cal - 1.0) * torch.log(1.0 - dt_cls)

            if self.usefocal:
                if torch.cuda.is_available():
                    alpha = torch.ones(dt_cls.shape).to(self.device) * self.alpha
                else:
                    alpha = torch.ones(dt_cls.shape) * self.alpha

                alpha = torch.where(torch.eq(assign_result_cal, 1.), alpha, 1. - alpha)
                focal_weight = torch.where(torch.eq(assign_result_cal, 1.), 1 - dt_cls, dt_cls)
                focal_weight = alpha * torch.pow(focal_weight, self.gamma)
                cls_fcloss_ib = focal_weight * cls_loss_ib
            else:
                cls_fcloss_ib = cls_loss_ib

            cls_loss.append(cls_fcloss_ib.sum() / positive_idx_box.sum())

            # bbox loss
            anch_w_box = self.anch_w[positive_idx_box]
            anch_h_box = self.anch_h[positive_idx_box]
            anch_x_box = self.anch_x[positive_idx_box]
            anch_y_box = self.anch_y[positive_idx_box]

            ann_w = assigned_anns[:, 2]
            ann_h = assigned_anns[:, 3]
            ann_x = assigned_anns[:, 0] + ann_w * 0.5
            ann_y = assigned_anns[:, 1] + ann_h * 0.5

            target_w = torch.log(ann_w / anch_w_box)
            target_h = torch.log(ann_h / anch_h_box)
            target_x = (ann_x - anch_x_box) / anch_w_box
            target_y = (ann_y - anch_y_box) / anch_h_box

            targets = torch.stack((target_x, target_y, target_w, target_h))

            box_loss_ib = torch.abs(targets - dt[ib, 0:4, positive_idx_box])

            # smooth l1 loss
            box_regression_loss_ib = torch.where(
                torch.le(box_loss_ib, 1.0 / 9.0),
                0.5 * 9.0 * torch.pow(box_loss_ib, 2),
                box_loss_ib - 0.5 / 9.0)

            ## l2 loss
            #box_regression_loss_ib = torch.pow(box_loss_ib, 2)

            bbox_loss.append(box_regression_loss_ib.sum() / positive_idx_box.sum())

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        bbox_loss = bbox_loss.sum()
        cls_loss = cls_loss.sum()
        loss = torch.add(bbox_loss,cls_loss)
        return loss/self.batch_size
