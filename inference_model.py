import numpy as np

from models.resnet import resnet50
from models.yolo import YOLOv3
from data.eval import model_inference_coconp
from data.trandata import CocoDataset, Resizer, Normalizer
from training.running import model_load_gen
from torchvision.transforms import Compose

dataset = CocoDataset("CrowdHuman/annotation_val_coco_style.json", "CrowdHuman/Images_val",
                      transform=Compose([Normalizer(), Resizer()]))

model = YOLOv3(numofclasses=1, backbone=None, istrainig=False)
model = model_load_gen(model, "70E_8B_800_1024_darknet53_E35",parallel_trained=False)
model = model.to(1)

result = model_inference_coconp(dataset, model)

np.save('CrowdHuman/70E_8B_800_1024_Darknet53_E35_0.7.npy', result)