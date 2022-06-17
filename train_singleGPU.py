from torchvision import transforms
from data.trandata import CocoDataset, OD_default_collater, Augmenter, Normalizer, Resizer
from torch.utils.data import DataLoader
import training.running as run

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg

def training_single(config=cfg):
    endepoch = 20
    dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train",
                          transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))
    loader = DataLoader(dataset, batch_size=config.batch_size,
                        shuffle=True, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=resnet50,
                   config=config).to(config.pre_device)

    run.training(model, loader, _cfg=config, logname="test_new_dataset",ending_epoch=endepoch)

def training_single_checkpoint(config=cfg):
    startepoch = 40
    endepoch = 60
    file = '70E_4B_800_1024_resnet50_4nd_continue_gpu0_E40'

    dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train",
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
    training_single(cfg)