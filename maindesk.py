import torch
from models.yolo import YOLOv3

model = YOLOv3()
test = torch.rand(10, 3, 256, 256)
f1, f2 ,f3 = model(test)
print(f1.shape, f2.shape, f3.shape)