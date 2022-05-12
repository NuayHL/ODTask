#各种各样的测试都在这里
import os
print(os.getcwd())
os.chdir("../ODTask")

import cv2
from util.visualization import show_bbox
from data.trandata import CrowdHDataset, OD_default_collater
from torch.utils.data import DataLoader
import numpy as np
import torch
from models.yolo import Yolov3_core
from training.assign import AnchAssign
from training.config import cfg

ID = 1

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=OD_default_collater)
assign_fun = AnchAssign()

for batch in loader:
    result = assign_fun.assign(batch["anns"])
    break

print(result)
print(torch.where(result>=1))








