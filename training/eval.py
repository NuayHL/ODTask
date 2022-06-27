import numpy as np
import cv2
import sys
import json
import torch
from torch import nn as nn
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from util.visualization import show_bbox
from copy import deepcopy
from data.dataset import CocoDataset, OD_default_collater
from training.config import cfg
from util.primary import progressbar, DDPsavetoNormal
from time import time

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

    def to_evaluation(self, img_id):
        '''
        return img_id x1y1wh score class
        '''
        self.img_id = img_id
        img_id = np.ones(shape=(self.len, 1),dtype=np.float32)*img_id
        result = np.concatenate((img_id, self._x1y1x2y2_to_x1y1wh()), 1)
        result[:, 6] = result[:, 6] + 1
        return result

    def _x1y1x2y2_to_x1y1wh(self):
        'return a x1y1wh style result'
        output = np.copy(self.result)
        output[:,2] = output[:,2] - output[:,0]
        output[:,3] = output[:,3] - output[:,1]
        return output

def coco_eval(model, val_dataset:CocoDataset, result_name='Default',logname='',
              logpath="trainingLog/", result_path="CrowdHuman/",config=cfg, device=None, resultnp=None):
    start = time()
    # Compute the inference result when it is not given
    if device is None:
        device = config.pre_device
    if resultnp is None:
        model = model.to(device)
        model.eval()
        resultnp = model_inference_coconp(val_dataset, model, config=config, device=device)
        np.save(result_path + result_name + '.npy', resultnp)
        print("result .npy saved")

    # Print the evaluation result to the log
    ori_std = sys.stdout
    with open(logpath+logname+".txt","a") as f:
        sys.stdout = f
        print("[["+result_name+"]]:")
        gt = val_dataset.annotations
        dt = gt.loadRes(resultnp)
        eval = COCOeval(gt, dt, 'bbox')
        eval.evaluate()
        eval.accumulate()
        eval.summarize()
        print("eval_times:%.2fs"%(time()-start))
        print("\n")
    sys.stdout = ori_std

def model_inference_coconp(dataset:CocoDataset, model, config=cfg, device=None):
    """
    return a result np.ndarray for COCOeval
    formate: imgidx x1y1wh score class
    """
    assert model.training is False,'Model should be set as evaluation states'
    if device is None:
        device = config.pre_device
    loader = DataLoader(dataset,shuffle=False,batch_size=config.batch_size, collate_fn=OD_default_collater)
    model.eval()
    result_list = []
    result_np = np.ones((0,7), dtype=np.float32)
    lenth = len(loader)
    print('Starting inference.')
    for idx, batch in enumerate(loader):
        if torch.cuda.is_available():
            imgs = batch['imgs'].to(device)
        else:
            imgs = batch['imgs']
        result_list_batch = model(imgs)
        result_list += result_list_batch
        progressbar(float((idx+1)/lenth),barlenth=40)
    print('Sorting...')
    lenth = len(result_list)
    for idx, result in enumerate(result_list):
        if result is not None:
            img_id = dataset.image_id[idx]
            ori_img = dataset.annotations.loadImgs(img_id)[0]
            fx = ori_img['width']/config.input_width
            fy = ori_img['height']/config.input_height
            result_formated = result.to_evaluation(img_id)
            result_formated[:, 1] *= fx
            result_formated[:, 3] *= fx
            result_formated[:, 2] *= fy
            result_formated[:, 4] *= fy
            result_np = np.concatenate((result_np,result_formated),0)
        else:
            pass
        progressbar(float((idx + 1) / lenth), barlenth=40)
    return result_np

def model_eval_loss(model, pthfilename, dataset, batchsize=4, device=cfg.pre_device, pararllel_trained=False):
    '''
    draw loss for testing dataset from the stored .pth model dict
    Please do not use this, it is meaning less.
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
    print("evluation complete:%.2f s"%(time()-starttime))
    print(pthfilename+'loss:', losses / lenth)

def inference_dataset_visualization(dataset:CocoDataset, sign, model, config=cfg):
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
    use to inference img from an outside image file.
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
    show_bbox(ori_img, bboxes, type='x1y1x2y2', color=[255,255,255], score=scores, thickness=thickness)


def model_save_gen(model:nn.Module, filename, last_epoch, optimizer=None, scheduler=None, path="models/model_pth"):
    save_dict = {"GEN":model.state_dict()}
    save_dict["last_epoch"] = last_epoch
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    torch.save(save_dict, path+"/"+filename+".pt")


def model_load_gen(filename, starting_epoch, model:nn.Module, optimizer=None, scheduler=None,
                   path="models/model_pth", parallel_trained=False):
    """
    return model, optimizer, scheduler, last_epoch
    """
    state_dict = torch.load(path+"/"+filename)
    if parallel_trained:
        model_dict = DDPsavetoNormal(state_dict["GEN"])
    else:
        model_dict = state_dict["GEN"]
    model.load_state_dict(model_dict, strict=True)
    if optimizer is not None:
        try: optimizer.load_state_dict(state_dict["optimizer"], strict=True)
        except:
            print("Optimizer Dict not FOUND or MISMATCH!")

    if scheduler is not None:
        try: scheduler.load_state_dict(state_dict["scheduler"], strict=True)
        except:
            print("Scheduler Dict not FOUND or MISMATCH!")

    if "last_epoch" in state_dict.keys():
        last_epoch = state_dict["last_epoch"]
    else: last_epoch = starting_epoch

    return model, optimizer, scheduler, last_epoch
