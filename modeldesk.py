import torch
from models.resnet import resnet50
from models.darknet53 import Darknet53
from util.visualization import draw_loss

draw_loss("70E_2B_800*1024_gpu0.txt","DarknetFirst")
draw_loss("70E_2B_800*1024_resnet50_gpu0.txt","ResnetFirst")

#pre_trained_dict = torch.load("models/model_pth/resnet18-5c106cde.pth")

# model = resnet50(1,pretrained=True).cuda()
#
# model2 = Darknet53(1).cuda()
#
# x = torch.rand((2,3,640,640)).cuda()
#
# f1,f2,f3 = model.yolo_extract(x)
#
# print(f1.shape, f2.shape, f3.shape)
#
# f1,f2,f3 = model2.yolo_extract(x)
#
# print(f1.shape, f2.shape, f3.shape)