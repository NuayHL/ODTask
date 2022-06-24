from torchvision import transforms

import training.eval
from data.dataset import CocoDataset, OD_default_collater, Augmenter, \
    Normalizer, Resizer, MixCocoDatset
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg

def training_single(config=cfg):
    dataset1 = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                           "CrowdHuman/Images_train")
    dataset2 = CocoDataset("WiderPerson/widerperson_all_coco_style.json",
                           "WiderPerson/Images", bbox_type="bbox")

    mixdataset = MixCocoDatset([dataset1, dataset2],
                               transform=transforms.Compose([Normalizer(),
                                                         Augmenter(),
                                                         Resizer()]))
    valdataset = CocoDataset("CrowdHuman/annotation_val_fbox_coco_style.json",
                             "CrowdHuman/Images_val",
                             bbox_type="bbox")
    print(len(mixdataset))
    loader = DataLoader(mixdataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=None,
                   config=config).to(config.pre_device)

    run.training(model, loader, valdataset=valdataset, _cfg=config, logname="Darknet53NoFocal_widerperson")

def training_single_checkpoint(config=cfg):
    startepoch = 5
    endepoch = 80
    file = '100E_16B_608_608_Darknet53NoFocal_widerperson_E5'

    dataset1 = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                           "CrowdHuman/Images_train")
    dataset2 = CocoDataset("WiderPerson/widerperson_all_coco_style.json",
                           "WiderPerson/Images",bbox_type="bbox")

    mixdataset = MixCocoDatset([dataset1, dataset2],
                               transform=transforms.Compose([Normalizer(),
                                                         Augmenter(),
                                                         Resizer()]))
    valdataset = CocoDataset("CrowdHuman/annotation_val_fbox_coco_style.json",
                             "CrowdHuman/Images_val",
                             bbox_type="bbox")
    loader = DataLoader(mixdataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=None,
                   config=config)
    model = training.eval.model_load_gen(model, file, parallel_trained=False)
    model = model.to(config.pre_device)

    run.training(model, loader, _cfg=config, valdataset=valdataset, logname="Darknet53NoFocal_widerperson",
                 starting_epoch=startepoch, ending_epoch=endepoch)

if __name__ == "__main__":
    training_single_checkpoint(cfg)