import numpy as np
from util.visualization import show_bbox
from data.trandata import CocoDataset, Resizer
from data.eval import inference_dataset_visualization
from models.resnet import resnet50
from models.yolo import YOLOv3
from training.running import model_load_gen
from torchvision.transforms import Compose
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from training.config import cfg

config = cfg
id = 1010

gtnp = np.load('CrowdHuman/70E_8B_800_1024_Darknet53_E35_0.7.npy')

gt = COCO("CrowdHuman/annotation_val_coco_style.json")

dt = gt.loadRes(gtnp)

eval = COCOeval(gt, dt, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()


