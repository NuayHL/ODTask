import numpy as np

from models.resnet import resnet50
from models.yolo import YOLOv3
from data.eval import model_eval
from data.trandata import CrowdHDataset, Resizer, Normalizer
from training.running import model_load_gen
from torchvision.transforms import Compose

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json",type='train',
                        transform=Compose([Normalizer(), Resizer()]))

model = YOLOv3(numofclasses=1, backbone=resnet50, istrainig=False)
model = model_load_gen(model, "70E_8B_800_1024_resnet50_4nd_E60",parallel_trained=False)
model = model.cuda()

result = model_eval(dataset, model)

np.save('CrowdHuman/70E_8B_800_1024_resnet50_4nd_E60_0.7_train.npy', result)