#用于对模型进行测试
import os
os.chdir("../ODTask")

from training.iou import IOU
import torch
a = torch.Tensor([[0,4,8,20]]).cuda()
b = torch.Tensor([[2,2,8,8]]).cuda()

iou = IOU()
print(iou(a,b))
