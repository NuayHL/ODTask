import numpy as np
from data.trandata import CocoDataset
from training.eval import coco_eval, model_load_gen
from models.yolo import YOLOv3
from training.config import cfg

config = cfg

val_dataset = CocoDataset("CrowdHuman/annotation_val_vbox_coco_style.json","CrowdHuman/Images_val")

model = YOLOv3(numofclasses=1, backbone=None)

model =  model_load_gen(model, "70E_8B_800_1024_darknet53_from55_E100").cuda()

coco_eval(model, val_dataset,
          logname="newtestlog",resultnp=None)



