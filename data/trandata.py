import json
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from copy import deepcopy
from util import progressbar
from torch.utils.data import Dataset
from torchvision import transforms
from training.config import cfg

# need to add logging module to print which kind input is used

preprocess_train = transforms.Compose([
    transforms.Normalize(mean=[0.46431773, 0.44211456, 0.4223358],
                         std=[0.29044453, 0.28503336, 0.29363019])
])

def odgt2coco(filepath, outputname, type):
    '''
    :param filepath: origin odgt file
    :param outputname: the name of output json file
    :param type: "train" or "val"
    :return: None

    About the CrowdHuman Dataset:
    CrowdHuman contains 15000 training imgs,
                        4370 validation imgs,
                        5000 testing imgs.

    About COCO format:{
        "info":{"year"}
        "images":[image]
        "annotations":[annotation]
        "categories":{}
    }
    image{
        "id": int,
        "width": int,
        "height": int,
        "file_name": str
    }
    annotation{
        "id": int,
        "image_id": int,
        "category_id": int,
        "bbox": [x,y,width,height] (vbox),
        "fbox": (fbox)
        "hbox": (hbox)
        "iscrowd": 0 or 1
    }
    categories[{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    assert type in ["train", "val"], "The type should be \'train\' or \'val\'"
    if type == "train": num_imgs = 15000
    else: num_imgs = 4370
    info = dict()
    images = []
    annotations = []
    categories = json.load(open("categories_coco.json"))

    info["year"] = 2018
    print("begin convert %s dataset annotations to coco format"%type)
    with open(filepath) as f:
        id = 0
        bbox_id=0
        for sample in f:
            id += 1
            img = json.loads(sample)
            w, h, _ = (cv2.imread("../CrowdHuman/images_"+type+"/"+img["ID"]+".jpg")).shape
            img_info = {"id":id, "width":w, "height":h, "file_name":img["ID"]}
            images.append(img_info)
            for bbox in img["gtboxes"]:
                if bbox["tag"] == "mask": continue
                if "ignore" in bbox["extra"].keys() and bbox["extra"]["ignore"] == 1: continue
                bbox_id += 1
                bbox_info={"id":bbox_id,"image_id":id,"category_id":1,"bbox":bbox["vbox"],"fbox":bbox["fbox"],"hbox":bbox["hbox"],"iscrowd":0}
                if "ignore" in bbox["head_attr"].keys() and bbox["head_attr"]["ignore"] == 1: del bbox_info["hbox"]
                annotations.append(bbox_info)
            progressbar(float(id/num_imgs))

    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open("../CrowdHuman/"+outputname+".json",'w'))

class CrowdHDataset(Dataset):
    def __init__(self, annotationPath, type="train", bbox_type=cfg.input_bboxtype, transform=None):
        super(CrowdHDataset, self).__init__()
        self.jsonPath = annotationPath
        self.imgPath = "CrowdHuman/Images_"+type+"/"
        self.type = type
        self.annotations = COCO(annotationPath)
        self.bbox_type = bbox_type
        self.transform = transform

    def __len__(self):
        return len(self.annotations.imgs)

    def __getitem__(self, idx):
        '''
        base output:
            sample['img'] = whc np.int32? img
            sample['anns] = n4 np.int32 img
        '''
        idx += 1
        img = self.annotations.loadImgs(idx)

        img = cv2.imread(self.imgPath + img[0]["file_name"] + ".jpg")
        img = img[:,:,::-1]

        anns = self.annotations.getAnnIds(idx)
        anns = deepcopy(self.annotations.loadAnns(anns))
        finanns = []
        for ann in anns:
            if self.bbox_type not in ann.keys(): continue
            # append 1: add category
            ann[self.bbox_type].append(1)
            finanns.append(ann[self.bbox_type])
        finanns = np.array(finanns).astype(np.int32)
        sample = {"img":img, "anns":finanns}

        if self.transform:
            sample = self.transform(sample)

        return sample

    # temp mistake, need to update
    def single_batch_input(self, sign):
        if isinstance(sign, str):
            idx = self._getwithname(sign)
        else:
            idx = sign
        data = self[idx]
        data = [data]
        imgs = torch.stack(
            [preprocess_train(torch.from_numpy(np.transpose(s["img"] / 255, (2, 0, 1))).float()) for s in data])
        annos = [np.array(s["anns"]).astype(np.float32) for s in data]
        return {"imgs": imgs, "anns": annos}

    def _getwithname(self, str):
        for idx in range(len(self)):
            if self.annotations.imgs[idx+1]["file_name"] == str:
                return idx

class evalDataset(Dataset):
    '''
    used only for evaluation
    '''
    def __init__(self, annotationPath, bboxtype = cfg.input_bboxtype):
        super(evalDataset, self).__init__()
        self.annotation = COCO(annotationPath)

    def __len__(self):
        return len(self.annotation.imgs)

    def __getitem__(self, idx):
        pass

class CocoDataset(Dataset):
    def __init__(self):
        super(CocoDataset, self).__init__()
        pass

    def __getitem__(self, item):
        pass

    def __len__(self):
        pass

class Normalizer():
    def __init__(self):
        self.mean = np.array([0.46431773, 0.44211456, 0.4223358])
        self.std = np.array([0.29044453, 0.28503336, 0.29363019])
    def __call__(self, sample):
        img = sample['img']
        img = img.astype(np.float32)/255
        img = (img - self.mean)/self.std
        return {'img':img, 'anns':sample['anns']}

class Augmenter():
    def __call__(self, sample, filp_x=0.5):
        if np.random.rand() < filp_x:
            img, anns = sample["img"], sample["anns"]
            img = img[:,::-1,:]

            _, width, _ = img.shape
            anns[:, 0] = width - anns[:, 0] - anns[:, 2]

            sample = {'img':img, 'anns':anns}
        return sample

class Resizer():
    def __init__(self,config = cfg):
        self.width = cfg.input_width
        self.height = cfg.input_height
    def __call__(self, sample):
        img, anns = sample["img"], sample["anns"].astype(np.float32)
        fy = self.height / float(img.shape[0])
        fx = self.width / float(img.shape[1])
        anns[:, 0] = fx * anns[:, 0]
        anns[:, 2] = fx * anns[:, 2]
        anns[:, 1] = fy * anns[:, 1]
        anns[:, 3] = fy * anns[:, 3]
        img = cv2.resize(img, (self.width, self.height))
        return {'img':img, 'anns':anns}

def OD_default_collater(data):
    '''
    used in torch.utils.data.DataLaoder as collater_fn
    parse the batch_size data into dict
    {"imgs":List lenth B, each with np.float32 img
     "anns":List lenth B, each with np.float32 ann, annType: x1y1wh}
    '''
    imgs = torch.stack([torch.from_numpy(np.transpose(s["img"], (2, 0, 1))).float() for s in data])
    annos = [s["anns"] for s in data]
    return {"imgs":imgs, "anns":annos}

def load_single_inferencing_img(img):
    '''
    Used for inferencing one single img
    :param img:
        str: file path
        np.ndarray: W x H x C
        torch.Tensor: B x C x W x H
    :return: Input Tensor viewed as batch_size 1
    '''
    if isinstance(img,str):
        img = cv2.imread(img)
        img = img[:,:,::-1]

    elif isinstance(img,torch.Tensor):
        return img
    elif isinstance(img,np.ndarray):
        pass
    else:
        raise NotImplementedError("Unknown inputType")

    img = (cv2.resize(img.astype(np.float32), (cfg.input_height, cfg.input_width)))/255
    img = np.transpose(img,(2,0,1))
    img = preprocess_train(torch.from_numpy(img).float())
    img = torch.unsqueeze(img, dim=0)
    return img

if __name__ == '__main__':
    odgt2coco("../CrowdHuman/annotation_val.odgt", "annotation_val_coco_style", "val")
    odgt2coco("../CrowdHuman/annotation_train.odgt","annotation_train_coco_style","train")

