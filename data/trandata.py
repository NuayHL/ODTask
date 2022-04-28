import json
import os
import cv2
import pycocotools
from torch.utils.data import Dataset

def odgt2coco(filepath, outputname, type):
    '''
    :param filepath: origin odgt file
    :param outputname: the name of output json file
    :param type: "train" or "val"
    :return: None
    '''
    assert type in ["train", "val"], "The type should be \'train\' or \'val\'"
    info = dict()
    images = []
    annotations = []
    categories = json.load(open("categories.json"))

    info["year"] = 2018
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
                bbox_id += 1
                bbox_info={"id":bbox_id,"image_id":id,"category_id":1,"bbox":bbox["fbox"],"iscrowd":0}
                annotations.append(bbox_info)

    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open("../CrowdHuman/"+outputname+".json",'w'))

class CrowdHDataset(Dataset):
    def __init__(self,crowdhuman_path):
        super(CrowdHDataset, self).__init__()
        self.img

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    odgt2coco("../CrowdHuman/annotation_val.odgt", "annotation_val_coco_style", "val")
    odgt2coco("../CrowdHuman/annotation_train.odgt","annotation_train_coco_style","train")

