import os

import torch

'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann, x1y1wh}
'''

from torchvision import transforms
from data.trandata import CrowdHDataset, OD_default_collater, Augmenter, Normalizer, Resizer
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg

def training_single(config=cfg):
    dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json",
                            transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=resnet50,
                   config=config, pretrained=True).to(config.pre_device)

    run.training(model, loader, _cfg=config, logname="resnet50_test")

if __name__ == "__main__":
    training_single(cfg)