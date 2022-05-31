import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch

from models.anchor import generateAnchors
from training.assign import AnchAssign

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def printImg(img, title: str='', type = 0):
    if type == 0: plt.imshow(img)
    else: plt.imshow(img, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()

def show_bbox(img, bboxs=[], type="default",color=[0,0,255],**kwargs):
    img = _add_bbox_img(img, bboxs=bboxs, type=type,color=color,**kwargs)
    printImg(img)

def _add_bbox_img(img, bboxs=[], type="default",color=[0,0,255],**kwargs):
    '''
    :param img: str for file path/np.ndarray (w,h,c)
    :param bboxs: one or list
    :param type: bbox format
    :param color: red
    :param kwargs: related to cv2.rectangle
    :return: img with bbox
    '''
    assert type in ["default","diagonal","crowdhuman"],"the bbox format should be \'default\' or \'diagonal\' or \'crowdhuman\'"
    if isinstance(img, str): img = cv2.imread(img)
    bboxs = bboxs if _isArrayLike(bboxs) else [bboxs]
    for bbox in bboxs:
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        if type == "diagonal": a, b = (bbox[0],bbox[1]),(bbox[2],bbox[3])
        elif type == "crowdhuman": a, b = (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3])
        else: a, b = (bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)),(bbox[0]+int(bbox[2]/2),bbox[1]+int(bbox[3]/2))
        img = cv2.rectangle(img,a,b,color)
    return img

def dataset_inspection(dataset, imgid, anntype="crowdhuman"):
    sample = dataset[imgid]
    img = sample["img"].astype(np.int32)
    anns = sample["anns"]
    show_bbox(img, anns, type=anntype)

def dataset_assign_inspection(dataset, imgid):
    sample = dataset[imgid]
    img = sample["img"].astype(np.int32)
    anns = np.array(sample["anns"]).astype(np.int32)
    assign_visualization(img, anns)

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

def assign_visualization(img, anns, annsidx=None, anchors=generateAnchors(), assignresult=None):
    if isinstance(anchors, np.ndarray):
        anchors = torch.from_numpy(anchors).to()
    if assignresult==None:
        assignf = AnchAssign()
        assignresult = assignf.assign([anns])
        assignresult = torch.squeeze(assignresult)
    num_anns = len(anns)
    if annsidx is None:
        annsidx = int(num_anns/2)

    sp_idx = torch.where(assignresult==annsidx+1)
    sp_anch = anchors[sp_idx]
    img = _add_bbox_img(img, sp_anch, type="crowdhuman")
    img = _add_bbox_img(img, anns[annsidx], type="crowdhuman")
    printImg(img)
