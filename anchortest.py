import torch
from training.iou import IOU

iou = IOU()

a = torch.Tensor([[1,1,2,2],[2,2,3,3]])
b = torch.Tensor([[1,1,2,2],[2,2,3,3],[2,2,4,4]])

c= b[[1,1,1],:]
print(c)
print(c.shape)