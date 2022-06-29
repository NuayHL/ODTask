from models.yolo import YOLOv3
from training.config import cfg
from training.eval import inference_single_visualization, checkpoint_load, model_load
from models.resnet import resnet50

model = YOLOv3(numofclasses=1,backbone=None)
model = model_load("100E_16B_608_608_Darknet53NoFocal_E100.pt",model, parallel_trained=False)
model = model.to(cfg.pre_device)

img = 'Img/IMG_20190913_140616.jpg'

inference_single_visualization(img, model,thickness=2)



