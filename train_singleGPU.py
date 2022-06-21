from torchvision import transforms

import training.eval
from data.dataset import CocoDataset, OD_default_collater, Augmenter, Normalizer, Resizer
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg

def training_single(config=cfg):
    dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train",
                          transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))
    valdataset = CocoDataset("CrowdHuman/annotation_val_fbox_coco_style.json",
                             "CrowdHuman/Images_val",
                             bbox_type="bbox")
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=None,
                   config=config).to(config.pre_device)

    run.training(model, loader, valdataset=valdataset, _cfg=config, logname="Darknet53NoFocal")

def training_single_checkpoint(config=cfg):
    startepoch = 95
    endepoch = 100
    file = '70E_8B_800_1024_darknet53_from55_E95'

    dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train",
                          transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))
    valdataset = CocoDataset("CrowdHuman/annotation_val_fbox_coco_style.json",
                             "CrowdHuman/Images_val",
                             bbox_type="bbox")
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=None,
                   config=config)
    model = training.eval.model_load_gen(model, file, parallel_trained=False)
    model = model.to(config.pre_device)

    run.training(model, loader, _cfg=config, valdataset=valdataset, logname="darknet53_from55",
                 starting_epoch=startepoch, ending_epoch=endepoch)

if __name__ == "__main__":
    training_single(cfg)