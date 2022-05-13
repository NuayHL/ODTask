import json
import cv2
import numpy as np
import torch
from pycocotools.coco import COCO
from util import progressbar
from torch.utils.data import Dataset
from torchvision import transforms
from training.config import cfg

# need to add logging module to print which kind input is used

preprocess_train = transforms.Compose([

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
    def __init__(self, annotationPath, type="train", bbox_type=cfg.input_bboxtype):
        super(CrowdHDataset, self).__init__()
        self.jsonPath = annotationPath
        self.imgPath = "CrowdHuman/Images_"+type+"/"
        self.type = type
        self.annotations = COCO(annotationPath)
        self.bbox_type = bbox_type

    def __len__(self):
        return len(self.annotations.imgs)

    def __getitem__(self, idx):
        idx += 1
        img = self.annotations.loadImgs(idx)
        fx = cfg.input_height / float(img[0]["height"])
        fy = cfg.input_width / float(img[0]["width"])
        img = cv2.imread(self.imgPath + img[0]["file_name"] + ".jpg")
        img = cv2.resize(img, (cfg.input_height, cfg.input_width))

        anns = self.annotations.getAnnIds(idx)
        anns = self.annotations.loadAnns(anns)
        finanns = []
        for ann in anns:
            if self.bbox_type not in ann.keys(): continue
            finanns.append(self._resizeGt(ann[self.bbox_type],fx,fy))
        return {"img":img, "anns":finanns}

    def _resizeGt(self,bbox,fx,fy):
        # resize the gt bbox according to the img resize
        bbox[0] *= fx
        bbox[2] *= fx
        bbox[1] *= fy
        bbox[3] *= fy
        return bbox

def OD_default_collater(data):
    '''
    used in torch.utils.data.DataLaoder as collater_fn
    parse the batch_size data into dict
    {"imgs":List lenth B, each with np.float32 img
     "anns":List lenth B, each with np.float32 ann}
    '''
    imgs = torch.stack([torch.from_numpy(np.transpose(preprocess_train(s["img"].astype(np.float32)), (2, 0, 1))).float() for s in data])
    annos = [np.array(s["anns"]).astype(np.float32) for s in data]

    return {"imgs":imgs, "anns":annos}


if __name__ == '__main__':
    odgt2coco("../CrowdHuman/annotation_val.odgt", "annotation_val_coco_style", "val")
    odgt2coco("../CrowdHuman/annotation_train.odgt","annotation_train_coco_style","train")

