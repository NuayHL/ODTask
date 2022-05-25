#各种各样的测试都在这里
import os
print(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]  = '0,1'


'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann}
'''

import cv2
from util.visualization import show_bbox
from data.trandata import CrowdHDataset, OD_default_collater
from torch.utils.data import DataLoader
import training.running as run
import torch
import torch.nn as nn

from training.loss import Defaultloss
from models.yolo import YOLOv3
from training.assign import AnchAssign
from training.config import cfg

loss = Defaultloss()

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=OD_default_collater)
for batch in loader:
    anno = batch["anns"]
    dummyinput = torch.rand((4,6,67200),requires_grad=True)
    with torch.autograd.set_detect_anomaly(True):
        losses = loss(dummyinput,anno)
    break

losses.backward()