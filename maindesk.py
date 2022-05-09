#各种各样的测试都在这里
import cv2

from util.visualization import show_bbox
from data.trandata import CrowdHDataset
from torch.utils.data import DataLoader
import numpy as np
import torch
from models.yolo import Yolov3_core

ID = 56

model = Yolov3_core().to("cuda")
dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
mix = dataset[ID]
img = mix["img"]
cv2.imshow(" ",img)
cv2.waitKey()
img = np.transpose(img,(2,0,1)).astype(np.float32)
print(img.shape)
img = torch.from_numpy(img)
input = torch.unsqueeze(img,0).to("cuda")
print(input.shape)
f1,f2,f3 = model(input)

print(f1.shape,f2.shape,f3.shape)

# show_bbox(mix["img"], mix["anns"],type = "crowdhuman")






