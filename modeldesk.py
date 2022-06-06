import torch
import cv2
from util.visualization import printImg, show_bbox
from models.resnet import resnet50
from models.darknet53 import Darknet53
from data.trandata import CrowdHDataset, Augmenter, Normalizer, Resizer
from util.visualization import dataset_inspection, dataset_assign_inspection, draw_loss, draw_loss_epoch
from torchvision import transforms
from models.anchor import generateAnchors

id = 5

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")

dataset_inspection(dataset, id)

anchors = generateAnchors(singleBatch=True)

show_bbox(dataset[id]['img'],anchors,type="x1y1x2y2")

