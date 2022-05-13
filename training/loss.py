import torch
import torch.nn as nn
from training.config import cfg
from training.assign import AnchAssign
from models.anchor import generateAnchors

class Defaultloss(nn.Module):
    def __init__(self):
        super(Defaultloss, self).__init__()
        self.label_assignment = AnchAssign()
    def forward(self,dt,gt):
        classification_loss=[]
        bbox_loss=[]
        assign_result = self.label_assignment.assign(gt)
        for idx, ass in enumerate(assign_result):
            pass


