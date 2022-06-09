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
from data.trandata import CrowdHDataset, OD_default_collater, Resizer
from torch.utils.data import DataLoader
import training.running as run
import torch
import torch.nn as nn
import torchvision.transforms as trans

from training.loss import Defaultloss
from models.yolo import YOLOv3
from models.resnet import resnet50
from training.assign import AnchAssign
from training.config import cfg
from training.running import model_load_gen
from util.visualization import show_bbox
from data.eval import inference_dataset_visualization

id = 2555

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")

model = YOLOv3(numofclasses=1,backbone=resnet50)
model = model_load_gen(model, "70E_8B_800_1024_resnet50_4nd_n_E60",parallel_trained=False)
model = model.to(cfg.pre_device)

inference_dataset_visualization(dataset, id, model)


