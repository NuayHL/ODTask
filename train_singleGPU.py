from torchvision import transforms

import training.eval
from data.dataset import CocoDataset, OD_default_collater, Augmenter, \
    Normalizer, Resizer, MixCocoDatset
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.retinanet import RetinaNet
from models.resnet import resnet50, resnet101, resnet18
from models.initialize import seed_init
from training.config import cfg

def train_single(config=cfg, end_epoch=cfg.trainingEpoch, pth_file=None):
    startepoch = 0
    endepoch = end_epoch
    model = YOLOv3(numofclasses=1, backbone=resnet18)

    dataset1 = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                           "CrowdHuman/Images_train", ignored_input=config.use_ignored)
    # dataset2 = CocoDataset("WiderPerson/widerperson_all_coco_style.json",
    #                        "WiderPerson/Images",bbox_type="bbox")
    #
    # mixdataset = MixCocoDatset([dataset1, dataset2],
    #                            transform=transforms.Compose([Normalizer(),
    #                                                      Augmenter(),
    #                                                      Resizer()]))
    valdataset = CocoDataset("CrowdHuman/annotation_val_coco_style_608_608.json",
                             "CrowdHuman/Images_val",
                             bbox_type="bbox")
    loader = DataLoader(dataset1, batch_size=config.batch_size, num_workers=4,
                        shuffle=True, collate_fn=OD_default_collater)

    run.training(model, loader, _cfg=config, valdataset=valdataset, logname="yolo_resnet18_temp",
                 starting_epoch=startepoch, ending_epoch=endepoch, checkpth=pth_file, save_per_epoch=20)

if __name__ == "__main__":
    seed_init(2008)
    pth_file = "120E_8B_608_608_yolo_resnet18_test_E50.pt"
    train_single(cfg, pth_file=pth_file)