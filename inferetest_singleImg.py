from models.yolo import YOLOv3
from models.retinanet import RetinaNet
from training.config import cfg
from training.eval import inference_single_visualization, checkpoint_load, model_load
from models.resnet import resnet50

model = RetinaNet()
model = model_load("100E_8B_800_1024_Retinanet3nd_test_E15.pt",model, parallel_trained=False)
model = model.to(cfg.pre_device)

img = 'Img/IMG_20190913_140611.jpg'

inference_single_visualization(img, model,thickness=2)



