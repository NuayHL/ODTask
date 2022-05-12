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

ID = 131

a = np.array([[2,2,8,8],[10,4,15,25]])

annos = [a]

assign_fun = AnchAssign()

result = assign_fun.assign(annos)

print(torch.where(result>1))








