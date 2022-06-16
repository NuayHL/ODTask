import numpy as np
import cv2
import json
import torch
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from util.visualization import show_bbox
from copy import deepcopy
from data.trandata import CrowdHDataset, OD_default_collater
from training.config import cfg
from util.primary import progressbar
from training.running import model_load_gen
from time import time
'''

'''

class Results():
    '''
    result format:
        N x 6: np.ndarray
        6: x1y1x2y2 target_score class_index
    '''
    def __init__(self, result):
        if isinstance(result, torch.Tensor): result = result.numpy()
        self.result = result
        self.len = result.shape[0]

    def load_bboxes(self):
        '''
        return bboxes and related scores
        '''
        return self.result[:,:4], self.result[:, 4:]

    def to_evaluation(self, idx):
        self.img_id = idx
        img_id = np.ones(shape=(self.len, 1),dtype=np.float32)*idx
        result = np.concatenate((img_id, self._x1y1x2y2_to_x1y1wh()), 1)
        result[:, 6] = result[:, 6] + 1
        return result

    def _x1y1x2y2_to_x1y1wh(self):
        'return a x1y1wh style result'
        output = np.copy(self.result)
        output[:,2] = output[:,2] - output[:,0]
        output[:,3] = output[:,3] - output[:,1]
        return output

def model_eval_coco(dataset, model, config=cfg):
    '''
    return a result np.ndarray for COCOeval
    '''
    assert model.istraining is False,'Model should be set as evaluation states'
    loader = DataLoader(dataset,shuffle=False,batch_size=8, collate_fn=OD_default_collater)
    model.eval()
    result_list = []
    result_np = np.ones((0,7), dtype=np.float32)
    lenth = len(loader)
    print('Starting inference.')
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            imgs = batch['imgs'].to(config.pre_device)
        result_list_batch = model(imgs)
        result_list += result_list_batch
        progressbar(float((idx+1)/lenth),barlenth=40)
    print('Sorting...')
    lenth = len(result_list)
    for idx, result in enumerate(result_list):
        try:
            result_np = np.concatenate((result_np,result.to_evaluation(idx+1)),0)
        except:
            pass
        progressbar(float((idx + 1) / lenth), barlenth=40)
    return result_np

def model_eval_loss(model, pthfilename, dataset, batchsize=4, device=cfg.pre_device, pararllel_trained=False):
    '''
    draw loss for testing dataset from the stored .pth model dict
    '''
    loader = DataLoader(dataset, batch_size=batchsize, collate_fn=OD_default_collater)
    model = model_load_gen(model, filename=pthfilename, parallel_trained=pararllel_trained)
    model = model.to(device)

    model.eval()
    lenth = len(loader)
    print('loader lenth:',lenth)
    losses = 0

    starttime = time()
    for idx, batch in enumerate(loader):
        batch['imgs'] = batch['imgs'].to(device)
        loss = model(batch)
        loss = loss.item()
        losses += loss
        progressbar(float((idx + 1) / lenth), barlenth=50)
    print("evluation complete:",time()-starttime,'s')
    print(pthfilename+'loss:', losses / lenth)

def inference_dataset_visualization(dataset:CrowdHDataset, sign, model, config=cfg):
    '''
    use to inference img from a CrowdHdataset like dataset
    '''
    ori_img = dataset.original_img_input(sign)
    singlebatch = dataset.single_batch_input(sign)
    model.eval()
    start = time()
    result = model(singlebatch['imgs'])[0]
    print('Inference time:%.2f s'%(time()-start))
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

def inference_single_visualization(img:str, model, config=cfg, thickness=3):
    '''
    use to inference img from a outside image file.
    '''
    ori_img = cv2.imread(img)
    if not isinstance(ori_img,np.ndarray):
        raise FileNotFoundError
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
    show_bbox(ori_img, bboxes, type='x1y1x2y2', score=scores, thickness=thickness)

def average_precision():
    pass

def average_recall():
    pass






