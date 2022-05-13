import torch
import torch.nn as nn
from training.config import cfg
from training.assign import AnchAssign
from models.anchor import generateAnchors

class Defaultloss(nn.Module):
    def __init__(self,):
        super(Defaultloss, self).__init__()
    def forward(self,dt,an,gt):
        pass


