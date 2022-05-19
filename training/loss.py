import torch
import torch.nn as nn
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
        self.anchs = anchors
        self._pre_anchor()

    def _pre_anchor(self):
        if torch.cuda.is_available():
            self.anchs = self.anchs.cuda()
        self.anch_w = self.anchs[:,2] - self.anchs[:,0]
        self.anch_h = self.anchs[:,3] - self.anchs[:,1]
        self.anch_x = self.anchs[:,0] + 0.5 * self.anch_w
        self.anch_y = self.anchs[:,1] + 0.5 * self.anch_h

    def forward(self,dt,gt):
        '''
        The input dt should be the same format as assign result
        i.e.  Tensor: Batchsize X (1+4+classes) X samples
        '''

        bbox_loss=[]
        cls_loss=[]

        assign_result = self.label_assignment.assign(gt)
        dt[:, 0, :] = torch.clamp(dt[:, 0, :], 1e-4, 1.0 - 1e-4)

        for ib in range(dt.shape[0]):
            # cls loss
            positive_idx_cls = torch.ge(assign_result[ib],-0.1)
            assign_result_cal = torch.clamp(assign_result[ib],0,1)
            cls_loss_ib = positive_idx_cls * (-dt[ib,:,0] * torch.log(assign_result_cal) +
                                        (dt[ib,:,0] - 1.0) * torch.log(1.0 - assign_result_cal))
            cls_loss.append(cls_loss_ib.sum()/positive_idx_cls.sum())

            # bbox loss
            imgAnn = gt[ib]
            imgAnn = torch.from_numpy(imgAnn).float()
            if torch.cuda.is_available():
                imgAnn = imgAnn.cuda()

            positive_idx_box = torch.ge(assign_result[ib]-1.0, -0.1)
            assign_result_bos = assign_result[ib][positive_idx_box].int()
            assigned_anns = imgAnn[assign_result_bos]
            anch_w_box = self.anch_w[positive_idx_box]
            anch_h_box = self.anch_h[positive_idx_box]
            anch_x_box = self.anch_x[positive_idx_box]
            anch_y_box = self.anch_y[positive_idx_box]

            ann_w = assigned_anns[:, 2] - assigned_anns[:, 0]
            ann_h = assigned_anns[:, 3] - assigned_anns[:, 1]
            ann_x = assigned_anns[:, 0] + ann_w * 0.5
            ann_y = assigned_anns[:, 1] + ann_h * 0.5

            target_w = torch.log(ann_w/anch_w_box)
            target_h = torch.log(ann_h/anch_h_box)
            target_x = (ann_x - anch_x_box) / anch_w_box
            target_y = (ann_y - anch_y_box) / anch_h_box

            targets = torch.stack((target_x, target_y, target_w, target_h))
            targets = targets.t()

            box_loss_ib = torch.abs(targets - dt[ib,positive_idx_box,1:5])
            box_loss_ib = torch.pow(box_loss_ib, 2)

            bbox_loss.append(box_loss_ib.sum()/positive_idx_box.sum())

        loss = 0
        for i in zip(bbox_loss, cls_loss):
            loss += (i[0]+i[1])

        return loss