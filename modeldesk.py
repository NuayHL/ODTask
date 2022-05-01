#用于对模型进行测试

from models.training import Iou
import torch
a = torch.Tensor([1,1,6,6]).cuda()
b = torch.Tensor([2,2,6,6]).cuda()

temp = Iou()
iou = temp.make()
print(iou(a,b))
