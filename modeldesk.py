import torch
from models.resnet import resnet50
from models.darknet53 import Darknet53
from data.trandata import CrowdHDataset
from util.visualization import dataset_inspection, dataset_assign_inspection

ID = 474
testset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
dataset_inspection(testset, ID)
