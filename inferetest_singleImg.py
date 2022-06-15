from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg
from training.running import model_load_gen
from data.eval import inference_single_visualization

model = YOLOv3(numofclasses=1,backbone=None)
model = model_load_gen(model, "70E_8B_800_1024_darknet53_E70",parallel_trained=False)
model = model.to(cfg.pre_device)

img = 'Img/IMG_20220610_154620.jpg'

inference_single_visualization(img, model,thickness=3)



