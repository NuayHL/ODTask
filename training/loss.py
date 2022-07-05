import math

import torch
import torch.nn as nn
import numpy as np
from training.config import cfg
from training.assign import AnchAssign
from models.anchor import generateAnchors

'''
super par:
    focal loss: alpha, gamma
    beta: balance between reg and cls loss
    cfg
'''

class FocalLoss_yoloInput(nn.Module):
    '''
    reg loss: smooth l1
    cls loss: bce + focal
    {"imgs":List lenth B, each with np.float32 img
     "anns":List lenth B, each with np.float32 ann}
    '''
    def __init__(self, assign_method=AnchAssign, anchors=generateAnchors(singleBatch=True), use_focal=True,
                 use_ignore=True, iou_type='ciou', device=None, config = cfg):
        super(FocalLoss_yoloInput, self).__init__()
        if isinstance(config, str):
            from .config import Config
            _cfg = Config(config)
        self.label_assignment = assign_method(config=config, device=device, using_ignored_input=use_ignore)
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
        self.beta = 2.0
        self.usefocal = use_focal
        self.useignore = use_ignore
        self.ioutype = iou_type
        if self.ioutype:
            print("Regloss using IoUloss type: %s"%self.ioutype)
            self.iouloss = IOUloss(iou_type=self.ioutype)
        else:
            print("Regloss using soft L1!")


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

        if self.useignore:
            assign_result, gt = self.label_assignment.assign(gt)
        else:
            assign_result = self.label_assignment.assign(gt)

        if torch.cuda.is_available():
            dt = dt.to(self.device)

        dt_class_md = torch.clamp(dt[:, 4:, :], 1e-7, 1.0 - 1e-7).clone()

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

            assigned_anns[:, 0] = assigned_anns[:, 0] + assigned_anns[:, 2] * 0.5
            assigned_anns[:, 1] = assigned_anns[:, 1] + assigned_anns[:, 3] * 0.5

            dt_reg_cal = dt[ib, 0:4, positive_idx_box]

            if self.ioutype:
                dt_reg_cal_wh = torch.clamp(dt_reg_cal[2:,:], max=50)
                dt_bbox_x = anch_x_box + anch_w_box * dt_reg_cal[0, :]
                dt_bbox_y = anch_y_box + anch_h_box * dt_reg_cal[1, :]
                dt_bbox_w = anch_w_box * torch.exp(dt_reg_cal_wh[0, :])
                dt_bbox_h = anch_h_box * torch.exp(dt_reg_cal_wh[1, :])
                dt_bbox = torch.stack([dt_bbox_x, dt_bbox_y, dt_bbox_w, dt_bbox_h])
                box_regression_loss_ib = self.iouloss(dt_bbox, assigned_anns.t())

            else:
                target_w = torch.log(assigned_anns[:, 2] / anch_w_box)
                target_h = torch.log(assigned_anns[:, 3] / anch_h_box)
                target_x = (assigned_anns[:, 0] - anch_x_box) / anch_w_box
                target_y = (assigned_anns[:, 1] - anch_y_box) / anch_h_box

                targets = torch.stack((target_x, target_y, target_w, target_h))

                box_loss_ib = torch.abs(targets - dt_reg_cal)

                # smooth l1 loss
                box_regression_loss_ib = torch.where(
                    torch.le(box_loss_ib, 1.0 / 9.0),
                    0.5 * 9.0 * torch.pow(box_loss_ib, 2),
                    box_loss_ib - 0.5 / 9.0)

            bbox_loss.append(box_regression_loss_ib.sum() / positive_idx_box.sum())

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        bbox_loss = bbox_loss.sum()
        cls_loss = cls_loss.sum()
        loss = torch.add(bbox_loss,cls_loss * self.beta)
        return loss/self.batch_size

class FocalLoss_splitInput(nn.Module):
    '''
    reg loss: smooth l1
    cls loss: bce + focal
    {"imgs":List lenth B, each with np.float32 img
     "anns":List lenth B, each with np.float32 ann}
    '''
    def __init__(self, assign_method=AnchAssign, anchors=generateAnchors(singleBatch=True), use_focal=True,
                 use_ignore=True, iou_type='ciou', device=None, config = cfg):
        super(FocalLoss_splitInput, self).__init__()
        if isinstance(config, str):
            from .config import Config
            _cfg = Config(config)
        self.label_assignment = assign_method(config=config, device=device, using_ignored_input=use_ignore)
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        self.anchs = anchors
        self.iouloss = IOUloss(iou_type= iou_type, reduction='sum')
        if device is None:
            self.device = config.pre_device
        else:
            self.device = device

        self._pre_anchor()
        self.alpha = 0.25
        self.gamma = 2.0
        self.usefocal = use_focal
        self.useignore = use_ignore

    def _pre_anchor(self):
        if torch.cuda.is_available():
            self.anchs = self.anchs.to(self.device)
        self.anch_w = self.anchs[:, 2] - self.anchs[:, 0]
        self.anch_h = self.anchs[:, 3] - self.anchs[:, 1]
        self.anch_x = self.anchs[:, 0] + 0.5 * self.anch_w
        self.anch_y = self.anchs[:, 1] + 0.5 * self.anch_h

    def forward(self, cls_dt, reg_dt, gt):
        '''
        dt: The input dt should be the same format as assign result
            i.e.  Tensor: Batchsize X (4+1+classes) X samples
        gt: list, each is np.array, with shape (4+1) at dim=-1
        '''
        self.classes = cls_dt.shape[1]
        self.batch_size = cls_dt.shape[0]

        bbox_loss = []
        cls_loss = []

        if self.useignore:
            assign_result, gt = self.label_assignment.assign(gt)
        else:
            assign_result = self.label_assignment.assign(gt)

        cls_dt = torch.clamp(cls_dt, 1e-7, 1.0 - 1e-7)

        if torch.cuda.is_available():
            cls_dt = cls_dt.to(self.device)
            reg_dt = reg_dt.to(self.device)

        # positive: exclude ignored sample
        # assigned: positive sample
        for ib in range(self.batch_size):
            positive_idx_cls = torch.ge(assign_result[ib], -0.1)
            # the not ignored ones
            positive_idx_box = torch.ge(assign_result[ib] - 1.0, -0.1)
            # the assigned ones
            debug_sum_po = positive_idx_box.sum()

            imgAnn = gt[ib]
            if not self.useignore:
                imgAnn = torch.from_numpy(imgAnn).float()
                if torch.cuda.is_available():
                    imgAnn = imgAnn.to(self.device)

            assign_result_box = assign_result[ib][positive_idx_box].long()-1
            target_anns = imgAnn[assign_result_box]

            # cls loss
            one_hot_bed = torch.zeros((assign_result.shape[1], self.classes), dtype=torch.int64)
            if torch.cuda.is_available():
                one_hot_bed = one_hot_bed.to(self.device)

            one_hot_bed[positive_idx_box, target_anns[:, 4].long() - 1] = 1

            assign_result_cal = one_hot_bed[positive_idx_cls]
            debug_sum_ = assign_result_cal.sum()
            cls_dt_cal = cls_dt[ib, :, positive_idx_cls].t()

            cls_loss_ib = - assign_result_cal * torch.log(cls_dt_cal) + \
                           (assign_result_cal - 1.0) * torch.log(1.0 - cls_dt_cal)

            debug_sum_ib = cls_loss_ib.sum()

            if self.usefocal:
                if torch.cuda.is_available():
                    alpha = torch.ones(cls_dt_cal.shape).to(self.device) * self.alpha
                else:
                    alpha = torch.ones(cls_dt_cal.shape) * self.alpha

                alpha = torch.where(torch.eq(assign_result_cal, 1.), alpha, 1. - alpha)
                focal_weight = torch.where(torch.eq(assign_result_cal, 1.), 1 - cls_dt_cal, cls_dt_cal)
                focal_weight = alpha * torch.pow(focal_weight, self.gamma)
                debug_sum_fo = focal_weight.sum()
                cls_fcloss_ib = focal_weight * cls_loss_ib
            else:
                cls_fcloss_ib = cls_loss_ib

            cls_loss.append(cls_fcloss_ib.sum() / positive_idx_box.sum())

            # bbox loss
            anch_w_box = self.anch_w[positive_idx_box]
            anch_h_box = self.anch_h[positive_idx_box]
            anch_x_box = self.anch_x[positive_idx_box]
            anch_y_box = self.anch_y[positive_idx_box]

            reg_dt_assigned = reg_dt[ib , :, positive_idx_box]

            dt_bbox_x = anch_x_box + reg_dt_assigned[0, :] * anch_w_box
            dt_bbox_y = anch_y_box + reg_dt_assigned[1, :] * anch_h_box
            reg_dt_assigned_wh = torch.clamp(reg_dt_assigned[2:, :], max=50)
            dt_bbox_w = anch_w_box * torch.exp(reg_dt_assigned_wh[0, :])
            dt_bbox_h = anch_h_box * torch.exp(reg_dt_assigned_wh[1, :])

            dt_bbox = torch.stack([dt_bbox_x, dt_bbox_y, dt_bbox_w, dt_bbox_h])

            target_anns[:, 0] += 0.5 * target_anns[:, 2]
            target_anns[:, 1] += 0.5 * target_anns[:, 3]

            box_regression_loss_ib = self.iouloss(dt_bbox, target_anns.t())

            bbox_loss.append(box_regression_loss_ib/ positive_idx_box.sum())

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        bbox_loss = bbox_loss.sum()
        cls_loss = cls_loss.sum()
        print('cls loss:%.8f'%cls_loss, 'bbox loss:%.4f'%bbox_loss)
        loss = torch.add(bbox_loss,cls_loss)
        return loss/self.batch_size

class IOUloss:
    """ Calculate IoU loss.
        based on https://github.com/meituan/YOLOv6/blob/main/yolov6/models/loss.py
    """
    def __init__(self, bbox_type='xywh', iou_type='ciou', reduction='none', eps=1e-7):
        """ Setting of the class.
        Args:
            bbox_type: (string), must be one of 'xywh' or 'xyxy'.
            iou_type: (string), can be one of 'ciou', 'diou', 'giou' or 'siou'
            reduction: (string), specifies the reduction to apply to the output, must be one of 'none', 'mean','sum'.
            eps: (float), a value to avoid devide by zero error.
        """
        self.box_format = bbox_type
        self.iou_type = iou_type.lower()
        self.reduction = reduction
        self.eps = eps

    def __call__(self, box1, box2):
        """ calculate iou. box1 and box2 are torch tensor with shape [4, m] and [4, m].
        """
        if self.box_format == 'x1y1x2y2':
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]
        elif self.box_format == 'xywh':
            b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
            b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
            b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
            b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
        elif self.box_format == 'x1y1wh':
            b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
            b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
        else:
            raise RuntimeError("None support bbox_type: %s"%self.box_format)
        # Intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
                (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # Union Area
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1 * h1 + w2 * h2 - inter + self.eps
        iou = inter / union

        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if self.iou_type == 'giou':
            c_area = cw * ch + self.eps  # convex area
            iou = iou - (c_area - union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if self.iou_type == 'diou':
                iou = iou - rho2 / c2
            elif self.iou_type == 'ciou':
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2 / c2 + v * alpha)
        elif self.iou_type == 'siou':
            # SIoU Loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1 + b2_x2 - b1_x1 - b1_x2) * 0.5 + self.eps
            s_ch = (b2_y1 + b2_y2 - b1_y1 - b1_y2) * 0.5
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            rho_x = (s_cw / cw) ** 2
            rho_y = (s_ch / ch) ** 2
            gamma = angle_cost - 2
            rho_x_g = torch.clamp(gamma * rho_x, max = 50)
            rho_y_g = torch.clamp(gamma * rho_y, max = 50)
            distance_cost = 2 - torch.exp(rho_x_g) - torch.exp(rho_y_g)
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            iou = iou - 0.5 * (distance_cost + shape_cost)
        loss = 1.0 - iou

        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()
        return loss

class BCEloss():
    def __init__(self, numofclass):
        self.numofclass = numofclass

    def __call__(self, cls_dt, cls_gt, assign_result):
        target_gt = cls_gt[assign_result]
        return 0

if __name__ == "__main__":
    box1 = torch.ones((4,10))*2
    box1[:2,:] -= 1
    box2 = torch.ones((4,10))
    box2[:2, :] -= 0.5

    iou = IOUloss(iou_type="iou", bbox_type='x1y1wh')
    print(iou(box1,box2))