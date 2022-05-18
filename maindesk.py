#各种各样的测试都在这里
import os
print(os.getcwd())
os.chdir("../ODTask")

'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann}
'''

import cv2
from util.visualization import show_bbox
from data.trandata import CrowdHDataset, OD_default_collater
from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from models.yolo import Yolov3_core
from models.backbone import Darknet53
from training.assign import AnchAssign
from training.config import cfg
from util.primary import numofParameters

ID = 1

model = Yolov3_core(numofclasses=1)
model = nn.DataParallel(model)
model = model.cuda()

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=OD_default_collater)
for batch in loader:
    print(batch["imgs"].shape)
    input = batch["imgs"].cuda()
    result = model(input)
    break

print(result[0].shape,result[1].shape,result[2].shape)

a  = torch.flatten(result[0],start_dim=2)
print(a.shape)








