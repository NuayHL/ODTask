from torchvision import transforms

import training.eval
from data.dataset import CocoDataset, OD_default_collater, Augmenter, \
    Normalizer, Resizer, MixCocoDatset
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.retinanet import RetinaNet
from models.resnet import resnet50
from training.config import cfg

def train_single(config=cfg, end_epoch=cfg.trainingEpoch, pth_file=None):
    startepoch = 0
    endepoch = end_epoch
    model = RetinaNet()

    dataset1 = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                           "CrowdHuman/Images_train", ignored_input=True)
    # dataset2 = CocoDataset("WiderPerson/widerperson_all_coco_style.json",
    #                        "WiderPerson/Images",bbox_type="bbox")
    #
    # mixdataset = MixCocoDatset([dataset1, dataset2],
    #                            transform=transforms.Compose([Normalizer(),
    #                                                      Augmenter(),
    #                                                      Resizer()]))
    valdataset = CocoDataset("CrowdHuman/annotation_val_coco_style_1024_800.json",
                             "CrowdHuman/Images_val",
                             bbox_type="bbox")
    loader = DataLoader(dataset1, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    run.training(model, loader, _cfg=config, valdataset=None, logname="test",
                 starting_epoch=startepoch, ending_epoch=endepoch, checkpth=pth_file)

if __name__ == "__main__":
    pth_file = "70E_8B_800_1024_darknet53_from55_E85.pt"
    train_single(cfg, pth_file=None)