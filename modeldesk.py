import torch
import cv2
from util.visualization import printImg
from models.resnet import resnet50
from models.darknet53 import Darknet53
from data.trandata import CrowdHDataset
from util.visualization import dataset_inspection, dataset_assign_inspection, draw_loss, draw_loss_epoch

img = cv2.imread("img1.jpg")
print(type(img))
img = img[:,:,::-1]
print(type(img))
printImg(img)