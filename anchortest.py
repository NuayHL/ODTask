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
from training.running import model_load_gen

img = cv2.imread("img2.jpg")

model = YOLOv3(numofclasses=1).to(cfg.pre_device)
model = model_load_gen(model, "testing")
model.eval()

result = model(img)
if result[0] == None:
    print("GG!")
else:
    print(result[0].load_bboxes)
