import numpy as np
import cv2
import json
import torch
from pycocotools.cocoeval import COCOeval
from util.visualization import show_bbox
from copy import deepcopy
from data.trandata import CrowdHDataset
from training.config import cfg
'''

'''

class Results():
    '''
    result format:
        N x 7: np.ndarray
        7: x1y1x2y2 target_score class_score class_index
    '''
    def __init__(self, result):
        if isinstance(result, torch.Tensor): result = result.numpy()
        self.result = result

    def load_bboxes(self):
        return self.result[:,:4], self.result[:, 5:]

def inference_dataset_visualization(dataset:CrowdHDataset, sign, model, config=cfg):
    ori_img = dataset.original_img_input(sign)
    singlebatch = dataset.single_batch_input(sign)
    model.eval()
    result = model(singlebatch['imgs'])[0]
    if result is None:
        print("gg!")
        return 0
    bboxes, scores = result.load_bboxes()
    fx = ori_img.shape[1]/float(config.input_width)
    fy = ori_img.shape[0]/float(config.input_height)
    bboxes[:, 0] *= fx
    bboxes[:, 2] *= fx
    bboxes[:, 1] *= fy
    bboxes[:, 3] *= fy
    show_bbox(ori_img, bboxes, type='x1y1x2y2', score=scores, thickness=1)

def inference_single_visualization(img:str, model, config=cfg):
    ori_img = cv2.imread(img)
    ori_img = ori_img[:,:,::-1]
    model.eval()
    result = model(img)
    result:Results = result[0]
    if result is None:
        print("gg!")
        return 0
    bboxes, scores = result.load_bboxes()
    fx = ori_img.shape[1]/float(config.input_width)
    fy = ori_img.shape[0]/float(config.input_height)
    bboxes[:, 0] *= fx
    bboxes[:, 2] *= fx
    bboxes[:, 1] *= fy
    bboxes[:, 3] *= fy
    show_bbox(ori_img, bboxes, type='x1y1x2y2', score=scores, thickness=3)

def average_precision():
    pass

def average_recall():
    pass






