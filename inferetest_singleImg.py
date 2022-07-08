from models.yolo import YOLOv3
from models.retinanet import RetinaNet
from training.config import cfg
from training.eval import inference_single_visualization, checkpoint_load, model_load
from models.resnet import resnet50, resnet18

model = YOLOv3(numofclasses=1,backbone=resnet18)
model = model_load("120E_8B_608_608_yolo_resnet18_test_E50.pt",model, parallel_trained=False)
model = model.to(cfg.pre_device)

img = 'Img/IMG_20190913_140616.jpg'

inference_single_visualization(img, model,thickness=2)



