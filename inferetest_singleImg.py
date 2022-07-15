from models.yolo import YOLOv3
from models.retinanet import RetinaNet
from training.config import cfg
from training.eval import inference_single_visualization, checkpoint_load, model_load
from models.resnet import resnet50, resnet18, resnet101

model = YOLOv3(numofclasses=1,backbone=None)
model = model_load("70E_8B_800_1024_darknet53_from55_E100.pt",model, parallel_trained=False)
model = model.to(cfg.pre_device)

img = 'Img/1066405,91c170005398a4a9.jpg'

inference_single_visualization(img, model,thickness=2)



