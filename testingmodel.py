import numpy as np
from torch.utils.data import DataLoader
from util.visualization import show_bbox
from util.primary import progressbar
from data.trandata import CocoDataset, Normalizer, Resizer, OD_default_collater
from data.eval import inference_dataset_visualization
from models.resnet import resnet50
from models.yolo import YOLOv3
from training.running import model_load_gen
from torchvision.transforms import Compose
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from training.config import cfg
from data.eval import model_eval_loss



if __name__ == '__main__':
    testset = CocoDataset('CrowdHuman/annotation_val_coco_style.json', "CrowdHuman/Images_val",
                          transform=Compose([Normalizer(), Resizer()]))
    #loss: 0.9178592605517362

    #trainset = CocoDataset('CrowdHuman/annotation_train_coco_style.json', type='train',
                            #transform=Compose([Normalizer(), Resizer()]))
    #loss: 0.21466813359703538

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=None)

    model_eval_loss(model, "70E_8B_800_1024_darknet53_E5", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E10", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E15", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E20", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E25", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E30", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E35", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E40", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E45", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E50", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_E55", testset)
