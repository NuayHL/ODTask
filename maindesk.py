#各种各样的测试都在这里
import os

import torch

print(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]  = '0,1'

'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann}
'''

import cv2
from data.trandata import CrowdHDataset, OD_default_collater
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.assign import AnchAssign
from training.config import cfg
import torch.distributed as dist


model = YOLOv3(numofclasses=1,istrainig=True, backbone=resnet50)
model = model.to(cfg.pre_device)

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
loader = DataLoader(dataset, batch_size=cfg.batch_size, collate_fn=OD_default_collater)
run.training(model,loader,logname="test1")
run.model_save_gen(model,"20E_4B_640*800")

