#用于对模型进行测试

from util.iou import Iou
import torch
a = torch.Tensor([1,1,2,2]).cuda()
b = torch.Tensor([1,1,2,2]).cuda()

temp = Iou()
iou = temp.make()
print(iou(a,b))
