#各种各样的测试都在这里
import os
print(os.getcwd())
os.chdir("../ODTask")

import cv2
from util.visualization import show_bbox
from data.trandata import CrowdHDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from models.yolo import Yolov3_core

ID = 131

model = Yolov3_core().to("cuda")
dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
loader = DataLoader(dataset, batch_size=8)
for idx, dict in loader:
    print(idx)
    print(dict)
    break










