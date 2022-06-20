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
    categories = json.load(open("data/categories_coco.json"))

    info["year"] = 2018
    print("begin convert %s dataset annotations to coco format"%type)
    with open(filepath) as f:
        id = 0
        bbox_id=0
        for sample in f:
            id += 1
            img = json.loads(sample)
            h, w, _ = (cv2.imread("CrowdHuman/Images_"+type+"/"+img["ID"]+".jpg")).shape
            img_info = {"id":id, "width":w, "height":h, "file_name":img["ID"]}
            images.append(img_info)
            for bbox in img["gtboxes"]:
                if bbox["tag"] == "mask": continue
                if "ignore" in bbox["extra"].keys() and bbox["extra"]["ignore"] == 1: continue
                bbox_id += 1
                area = bbox['vbox'][2] * bbox['vbox'][3] #vbox area
                bbox_info={"id":bbox_id,"image_id":id,"category_id":1,
                           "vbox":bbox["vbox"],"bbox":bbox["fbox"],"hbox":bbox["hbox"],
                           "area":area, "iscrowd":0}
                if "ignore" in bbox["head_attr"].keys() and bbox["head_attr"]["ignore"] == 1: del bbox_info["hbox"]
                annotations.append(bbox_info)
            progressbar(float(id/num_imgs))

    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open("CrowdHuman/"+outputname+".json",'w'))

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
        self.width = config.input_width
        self.height = config.input_height
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

class CocoDataset(Dataset):
    '''
    Two index system:
        using idx: search using the idx, the position which stored the image_id. Start from 0.
        using id: search using image_id. Usually starts from 1.
    Relation:
        Default: id = idx + 1
        Always right: id = self.image_id[idx] (adopted)
    '''
    def __init__(self, annotationPath, imgFilePath, bbox_type=cfg.input_bboxtype,
                 transform=transforms.Compose([Normalizer(), Resizer()])):
        super(CocoDataset, self).__init__()
        self.jsonPath = annotationPath
        self.imgPath = imgFilePath + "/"
        self.annotations = COCO(annotationPath)
        self.image_id = self.annotations.getImgIds()
        if bbox_type is None:
            self.bbox_type = bbox_type
        else:
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
        img = self.annotations.loadImgs(self.image_id[idx])

        img = cv2.imread(self.imgPath + img[0]["file_name"] + ".jpg")
        img = img[:,:,::-1]

        anns = self.annotations.getAnnIds(imgIds=self.image_id[idx])
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

    # return real size img
    def original_img_input(self,id):
        '''
        id: img_ID
        '''
        if isinstance(id, str):
            _, id = self._getwithname(id)
        img = self.annotations.loadImgs(id)
        img = cv2.imread(self.imgPath + img[0]["file_name"] + ".jpg")
        img = img[:,:,::-1]
        return img

    def single_batch_input(self, idx):
        '''
        idx: idx
        '''
        if isinstance(idx, str):
            idx, _ = self._getwithname(idx)
        data = self[idx]
        norm = Normalizer()
        resizer = Resizer()
        data = resizer(norm(data))
        return OD_default_collater([data])

    def _getwithname(self, str):
        '''
        return (idx, id)
        '''
        for idx in range(len(self)):
            if self.annotations.imgs[idx+1]["file_name"] == str:
                return idx, self.annotations.imgs[idx+1]["id"]
        raise KeyError('Can not find img with name %s'%str)


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

def load_single_inferencing_img(img, device=cfg.pre_device):
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
        if torch.cuda.is_available():
            return img.to(device)
        else:
            return img
    elif isinstance(img,np.ndarray):
        pass
    else:
        raise NotImplementedError("Unknown inputType")

    img = (cv2.resize(img.astype(np.float32), (cfg.input_width, cfg.input_height)))/255
    img = np.transpose(img,(2,0,1))
    img = preprocess_train(torch.from_numpy(img).float())
    img = torch.unsqueeze(img, dim=0)
    return img.to(device)

if __name__ == '__main__':
    odgt2coco("CrowdHuman/annotation_val.odgt", "annotation_val_fbox_coco_style", "val")
    #odgt2coco("CrowdHuman/annotation_train.odgt","annotation_train_coco_style","train")

