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

def training_single_checkpoint(config=cfg):
    startepoch = 40
    endepoch = 60
    file = '70E_4B_800_1024_resnet50_4nd_continue_gpu0_E40'

    dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json",
                            transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=resnet50,
                   config=config)
    model = run.model_load_gen(model, file, parallel_trained=True)
    model = model.to(config.pre_device)

    run.training(model, loader, _cfg=config, logname="resnet50_4nd_n",
                 starting_epoch=startepoch, ending_epoch=endepoch)

if __name__ == "__main__":
    training_single_checkpoint(cfg)