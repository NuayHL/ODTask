import numpy as np
from util.visualization import show_bbox
from data.trandata import CrowdHDataset, Resizer
from data.eval import inference_dataset_visualization
from models.resnet import resnet50
from models.yolo import YOLOv3
from training.running import model_load_gen
from torchvision.transforms import Compose
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

gtnp = np.load('CrowdHuman/70E_8B_800_1024_resnet50_4nd_E60_0.7_train.npy')
gtnp[:,6] += gtnp[:,6] + 1

gt = COCO('CrowdHuman/annotation_train_coco_style_area.json')
dt = gt.loadRes(gtnp)

eval = COCOeval(gt, dt, 'bbox')
eval.evaluate()
eval.accumulate()
eval.summarize()


