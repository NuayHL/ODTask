from models.resnet import resnet18, resnet101,resnet50, resnet34
from models.retinanet import RetinaNet
from models.yolo import YOLOv3
from util.primary import numofParameters

model= YOLOv3(numofclasses=1)
print(model)
print(numofParameters(model))
# model= RetinaNet(numofclass=1, backbone=resnet50)
# print(numofParameters(model))
# model= YOLOv3(numofclasses=1, backbone=resnet101)
# print(numofParameters(model))
# model= YOLOv3(numofclasses=1,backbone=resnet50)
# print(numofParameters(model))
# model= YOLOv3(numofclasses=1,backbone=resnet34)
# print(numofParameters(model))
# model= YOLOv3(numofclasses=1,backbone=resnet18)
# print(numofParameters(model))

import torch

dicts = torch.load("best_ckpt.pt")
# print(dicts.keys())