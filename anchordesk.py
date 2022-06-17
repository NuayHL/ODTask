import torch
import cv2
import numpy as np
from util.visualization import printImg, show_bbox, _add_bbox_img
from models.resnet import resnet50
from models.darknet53 import Darknet53
from data.trandata import CocoDataset, Augmenter, Normalizer, Resizer
from util.visualization import dataset_inspection, dataset_assign_inspection, draw_loss, draw_loss_epoch
from torchvision import transforms
from models.anchor import generateAnchors, anchors_parse
from training.config import cfg

gray_img = np.ones((1600,2048,3),dtype=np.int32)*191
gray_img[400:1200, 512:1024+512, :] -= 64

anchors = generateAnchors(singleBatch=True)
anchors[:, 0] += cfg.input_width / 2
anchors[:, 2] += cfg.input_width / 2
anchors[:, 1] += cfg.input_height / 2
anchors[:, 3] += cfg.input_height / 2
anchors = anchors.astype(np.int32)
parsed_anchors = anchors_parse()

gray_img = _add_bbox_img(gray_img, parsed_anchors[2][2], type='x1y1x2y2')
gray_img = _add_bbox_img(gray_img, parsed_anchors[2][0], color=[255,0,0], type='x1y1x2y2')
printImg(gray_img)
#show_bbox(gray_img, parsed_anchors[2][0],type='x1y1x2y2')

