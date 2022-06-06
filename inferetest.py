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
from data.eval import inference_single

id = 456

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")

model = YOLOv3(numofclasses=1,backbone=resnet50, istrainig=True)
model = model_load_gen(model, "70E_2B_800_1024_resnet50_3nd_gpu0_E70",parallel_trained=True)
model = model.to(cfg.pre_device)
model.train()
batch = dataset.single_batch_input("273271,1c76f000a7edecb7")
batch['imgs'] = batch['imgs'].to(cfg.pre_device)
loss = model(batch)
print(loss)



