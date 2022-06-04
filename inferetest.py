#各种各样的测试都在这里
import os
print(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]  = '0,1'

'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann}
'''

import cv2
import numpy as np
from util.visualization import show_bbox
from data.trandata import CrowdHDataset, OD_default_collater
from torch.utils.data import DataLoader
import training.running as run
import torch
import torch.nn as nn

from training.loss import Defaultloss
from models.yolo import YOLOv3
from models.resnet import resnet50
from training.assign import AnchAssign
from training.config import cfg
from training.running import model_load_gen
from util.visualization import show_bbox

img = cv2.imread("img1.jpg")

model = YOLOv3(numofclasses=1,backbone=resnet50).to(cfg.pre_device)
model = model_load_gen(model, "70E_2B_800_1024_resnet50_3nd_gpu0_E70",parallel_trained=True)
model.eval()

result = model(img)
if result[0] == None:
    print("GG!")
else:
    print(result[0].bboxes)
    bboxes = (result[0].bboxes).numpy().astype(np.int32)
    show_bbox("img1.jpg", bboxes, color=[255,0,0])
