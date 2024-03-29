from data.dataset import CocoDataset

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg
from training.eval import inference_dataset_visualization, model_load

id = 2555

dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json","CrowdHuman/Images_train",transform=None)
#dataset = CocoDataset("CrowdHuman/annotation_val_vbox_coco_style.json")

model = YOLOv3(numofclasses=1,backbone=None)
model = model_load("70E_8B_800_1024_darknet53_from55_E100.pt",model, parallel_trained=False)
model = model.to(cfg.pre_device)

inference_dataset_visualization(dataset, id, model)


