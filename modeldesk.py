#用于对模型进行测试
import os
os.chdir("../ODTask")

from training.iou import IOU
import torch
a = torch.Tensor([[1,1,2,2]]).cuda()
b = torch.Tensor([[1,1,2,2]]).cuda()

iou = IOU()
print(iou(a,b))
