import torch
from models.resnet import resnet50
from models.darknet53 import Darknet53
from data.trandata import CrowdHDataset
from util.visualization import dataset_inspection, dataset_assign_inspection, draw_loss, draw_loss_epoch

draw_loss_epoch("70E_2B_800_1024_resnet50_3nd_gpu1.txt",3750)
