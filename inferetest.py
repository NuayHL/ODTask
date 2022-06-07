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

model = YOLOv3(numofclasses=1,backbone=resnet50)
model = model_load_gen(model, "70E_2B_800_1024_resnet50_4nd_gpu0_E20",parallel_trained=True)
model = model.to(cfg.pre_device)
batch = dataset.single_batch_input(id)
img = batch['imgs'].to(cfg.pre_device)
batch['imgs'] = batch['imgs'].to(cfg.pre_device)
bboxs = inference_single(img, model)
show_bbox(dataset[id]['img'], bboxs, type ="x1y1x2y2")
model.istraining = True
model.train()
loss = model(batch)
print(loss)

