import json
import os
import cv2
import pycocotools
from torch.utils.data import Dataset

def odgt2coco(filepath, outputname):
    info = dict()
    images = []
    annotations = []
    categories = []
    with open(filepath) as f:
        pass #zaizheli
    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open("../CrowdHuman/"+outputname+".json"))

class CrowdHDataset(Dataset):
    def __init__(self,crowdhuman_path):
        super(CrowdHDataset, self).__init__()
        self.img

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass


if __name__ == '__main__':
    f = open('../CrowdHuman/annotation_train.odgt')

    for imggt in f:
        imggt = json.loads(imggt)
        create = open('../CrowdHuman/annotation_train/'+imggt['ID']+'.txt','w')
        for objects in imggt['gtboxes']:
            m = objects['tag']+' '+str(objects['fbox'])
            print(m,file=create)
        create.close()
    f.close()
