import json
import cv2
import numpy as np
from util.primary import progressbar

import os

'''
format in BGR! Need to change to RGB
'''

def cal_img_mean_std(coco_style_annotation_path, type):
    with open(coco_style_annotation_path) as f:
        annotation = json.load(f)
    assert "images" in annotation.keys(),"Unknown format of annotation. The annotation must have key:\'images\'"

    mean = np.zeros((3)).astype(np.float_)
    std = np.zeros((3)).astype(np.float_)
    imgnum = len(annotation["images"])
    num_pixels = 0
    for idx, imgs in enumerate(annotation["images"]):
        real_img = cv2.imread("CrowdHuman/Images_"+type+"/"+imgs["file_name"]+".jpg").astype(np.float_)
        num_pixels += imgs["width"] * imgs["height"]
        for i in range(3):
            mean[i] += real_img[:, :, i].sum()
            std[i] += np.power(real_img[:, :, i],2).sum()
        progressbar(float(idx/imgnum))

    mean = mean/(255*num_pixels)
    std = np.power(std/(255*255*num_pixels)-np.power(mean,2),0.5)

    return mean,std

if __name__ == "__main__":
    print(os.getcwd())

    mean, std = cal_img_mean_std("CrowdHuman/annotation_train_coco_style.json", "train")
    print("mean:", mean)
    #[0.4223358  0.44211456 0.46431773]
    print("std:", std)
    #[0.29363019 0.28503336 0.29044453]
