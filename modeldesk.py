#用于对模型进行测试

from models.yolo import YOLOv3
import torch
import time
from util.primary import numofParameters

DEVICE = "cuda"

a = torch.rand((3,3,512,512)).to(DEVICE)
model = YOLOv3().to(DEVICE)

print(numofParameters(model))
tic = time.time()
f1, f2, f3 = model(a)
print(time.time()-tic)
print(f1.shape, f2.shape, f3.shape)
