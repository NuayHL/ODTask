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
from models.yolo import Yolov3_core
from training.assign import AnchAssign
from training.config import cfg
from util.primary import numofParameters

ID = 1

testinput = torch.rand((8,3,512,512))

model = Yolov3_core(numofclasses=1).cuda()
print(numofParameters(model))
dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=OD_default_collater)
for batch in loader:
    print(batch["imgs"].shape)
    input = batch["imgs"].cuda()
    result = model(input)
    break

print(result[0].shape,result[1].shape,result[2].shape)








