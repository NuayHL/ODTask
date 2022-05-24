import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import one_hot
from training.config import cfg
from training.assign import AnchAssign
from models.anchor import generateAnchors


class Defaultloss(nn.Module):
    '''
    {"imgs":List lenth B, each with np.float32 img
     "anns":List lenth B, each with np.float32 ann}
    '''

    def __init__(self, assign_method=AnchAssign(), anchors=generateAnchors(singleBatch=True)):
        super(Defaultloss, self).__init__()
        self.label_assignment = assign_method
        if isinstance(anchors, np.ndarray):
            anchors = torch.from_numpy(anchors)
        self.anchs = anchors
        self._pre_anchor()

    def _pre_anchor(self):
        if torch.cuda.is_available():
            self.anchs = self.anchs.to(cfg.pre_device)
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
            dt = dt.to(cfg.pre_device)

        dt_class_md = torch.clamp(dt[:, 4:, :], 1e-4, 1.0 - 1e-4).clone()

        for ib in range(dt.shape[0]):

            positive_idx_cls = torch.ge(assign_result[ib], -0.1)
            # the not ignored ones
            positive_idx_box = torch.ge(assign_result[ib] - 1.0, -0.1)
            # the assigned ones

            total_pcls = positive_idx_cls.sum()
            total_pbox = positive_idx_box.sum()

            imgAnn = gt[ib]
            imgAnn = torch.from_numpy(imgAnn).float()
            if torch.cuda.is_available():
                imgAnn = imgAnn.to(cfg.pre_device)

            assign_result_box = assign_result[ib][positive_idx_box].long()-1
            assigned_anns = imgAnn[assign_result_box]

            # cls loss
            assign_result_cal = torch.clamp(assign_result[ib][positive_idx_cls], 0., 1.)
            one_hot_bed = torch.zeros((assign_result.shape[1], self.classes), dtype=torch.int64)
            if torch.cuda.is_available():
                one_hot_bed = one_hot_bed.to(cfg.pre_device)
            one_hot_bed[positive_idx_box] = one_hot(assigned_anns[:, 4].long() - 1,
                                                    num_classes=self.classes)
            assign_result_cal = torch.cat((torch.unsqueeze(assign_result_cal, dim=1),
                                          one_hot_bed[positive_idx_cls]), dim=1)
            assign_result_cal = torch.clamp(assign_result_cal,1e-4, 1.0 - 1e-4)
            dt_cls = dt_class_md[ib, :, positive_idx_cls].t()
            cls_loss_ib = -dt_cls * torch.log(assign_result_cal) + \
                          (dt_cls - 1.0) * torch.log(1.0 - assign_result_cal)
            cls_loss.append(cls_loss_ib.sum() / positive_idx_cls.sum())

            # bbox loss

            # skip if no bbox matched
            if total_pbox==0: continue

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
            box_loss_ib = torch.pow(box_loss_ib, 2)

            bbox_loss.append(box_loss_ib.sum() / positive_idx_box.sum())

        bbox_loss = torch.stack(bbox_loss)
        cls_loss = torch.stack(cls_loss)
        bbox_loss = bbox_loss.sum()
        cls_loss = cls_loss.sum()
        loss = torch.add(bbox_loss,cls_loss)
        return loss/self.batch_size
