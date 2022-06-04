import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.anchor import generateAnchors
from training.assign import AnchAssign

def _isArrayLike(obj):
    # not working for numpy
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

class DatasetVisual():
    def __init__(self, dataset):
        self.dataset = dataset
        self.assignresult = {}

def printImg(img, title: str='', type = 0):
    if type == 0: plt.imshow(img)
    else: plt.imshow(img, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()

def dataset_inspection(dataset, imgid, anntype="x1y1wh"):
    sample = dataset[imgid]
    img = sample["img"].astype(np.int32)
    anns = sample["anns"]
    show_bbox(img, anns, type=anntype)

def dataset_assign_inspection(dataset, imgid, annsidx=None):
    sample = dataset[imgid]
    img = sample["img"].astype(np.int32)
    anns = np.array(sample["anns"]).astype(np.int32)
    assign_visualization(img, anns, annsidx)

def assign_visualization(img, anns, annsidx=None, anchors=generateAnchors(singleBatch=True),
                         assignresult=None, anntype="x1y1wh"):
    '''
    :param img:
    :param anns:
    :param annsidx: choose an index refered to certain annotation bbox, default is the middle
    :param anchors: pre defined anchors
    :param assignresult: assign result
    :param anntype: anns bbox type
    :return: img with bbox
    '''
    if assignresult==None:
        assignf = AnchAssign()
        assignresult = assignf.assign([anns])
        assignresult = torch.squeeze(assignresult)
    num_anns = len(anns)
    if annsidx is None:
        annsidx = int(num_anns/2)
    assert annsidx>=0 and annsidx<num_anns, "invalid ann_index for these img, change a suitable \'annsidx\'"

    sp_idx = torch.eq(assignresult, annsidx+1).to("cpu")
    sp_anch = (anchors[sp_idx]).astype(np.int32)
    img = _add_bbox_img(img, sp_anch, type="x1y1x2y2")
    img = _add_bbox_img(img, [anns[annsidx,:]], type=anntype, color=[255,0,0], thickness=3, lineType=8)
    printImg(img)

def show_bbox(img, bboxs=[], type="xywh",color=[0,0,255],**kwargs):
    img = _add_bbox_img(img, bboxs=bboxs, type=type,color=color,**kwargs)
    printImg(img)

def _add_bbox_img(img, bboxs=[], type="xywh",color=[0,0,255],**kwargs):
    '''
    :param img: str for file path/np.ndarray (w,h,c)
    :param bboxs: one or lists or np.ndarray
    :param type: xywh, x1y1x2y2, x1y1wh
    :param color: red
    :param kwargs: related to cv2.rectangle
    :return: img with bbox
    '''
    assert type in ["xywh","x1y1x2y2","x1y1wh"],"the bbox format should be \'xywh\' or \'x1y1x2y2\' or \'x1y1wh\'"
    if isinstance(img, str):
        img = cv2.imread(img)
        img = img[:,:,::-1]
    if isinstance(bboxs, np.ndarray):
        assert len(bboxs.shape)==2 and bboxs.shape[1]>=4, "invalid bboxes shape for np.ndarray"
        bboxs = bboxs.astype(np.int8)
    bboxs = bboxs if _isArrayLike(bboxs) else [bboxs]
    for bbox in bboxs:
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        if type == "x1y1x2y2": a, b = (bbox[0],bbox[1]),(bbox[2],bbox[3])
        elif type == "x1y1wh": a, b = (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3])
        else: a, b = (bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)),(bbox[0]+int(bbox[2]/2),bbox[1]+int(bbox[3]/2))
        img = cv2.rectangle(img,a,b,color, **kwargs)
    return img

def draw_loss(file_name,outputImgName="loss",logpath="trainingLog",savepath="trainingLog/lossV"):
    with open(logpath+"/"+file_name,"r") as f:
        losses = f.readlines()
        loss_list = []
        index = []
        start_idx = 0
        for i in losses:
            if "WARNING" in i:
                continue
            loss = float(i[(i.rfind(":")+1):])
            loss_list.append(loss)
            index.append(start_idx)
            start_idx += 1
    fig, ax = plt.subplots()
    ax.plot(index, loss_list)
    ax.set(xlabel="Iteration(times)",ylabel="Loss",title="Training Loss for "+file_name)
    ax.grid()

    fig.savefig(savepath+"/"+file_name+"_"+outputImgName+".png")
    plt.show()

def draw_loss_epoch(file_name, num_in_epoch, outputImgName="loss per epoch", logpath="trainingLog", savepath="trainingLog/lossV"):
    with open(logpath+"/"+file_name,"r") as f:
        losses = f.readlines()
        epoch_loss = 0
        epoch_count = 0
        start_idx = 0
        loss_list = []
        index = []
        for i in losses:
            if "WARNING" in i:
                continue
            if start_idx % num_in_epoch == 0:
                index.append(epoch_count)
                loss_list.append(epoch_loss / num_in_epoch)
                epoch_loss = 0
                epoch_count += 1
            loss = float(i[(i.rfind(":")+1):])
            epoch_loss += loss
            start_idx += 1
        if start_idx % num_in_epoch != 0:
            index.append(epoch_count)
            loss_list.append(epoch_loss / (start_idx % num_in_epoch))
    index = index[1:]
    loss_list = loss_list[1:]
    fig, ax = plt.subplots()
    ax.plot(index, loss_list)
    ax.set(xlabel="Epochs",ylabel="Loss",title="Training Loss per epoch for "+file_name)
    ax.grid()

    fig.savefig(savepath+"/"+file_name+"_"+outputImgName+".png")
    plt.show()