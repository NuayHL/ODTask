import torch
import cv2
from util.visualization import printImg
from models.resnet import resnet50
from models.darknet53 import Darknet53
from data.trandata import CrowdHDataset
from util.visualization import dataset_inspection, dataset_assign_inspection, draw_loss, draw_loss_epoch

draw_loss("70E_8B_640_800_test.txt")