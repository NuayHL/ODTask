import os
import torch
from torchvision import transforms

import training.eval
import training.running as run
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from models.yolo import YOLOv3
from models.resnet import resnet50

from data.dataset import CocoDataset, OD_default_collater, Augmenter, Normalizer, Resizer
from torch.utils.data import DataLoader
from training.config import cfg


def training_process(rank, world_size, config):
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train",
                          transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))

    ddsampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=ddsampler, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=resnet50, config=config, pretrained=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    run.training(ddp_model, loader, logname="resnet50_cosR", scheduler='cosineRestarts')

def training_process_checkpoint(rank, world_size, config):
    startepoch = 20
    endepoch = 40
    file = '70E_2B_800_1024_resnet50_4nd_gpu0_E20'
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train",
                          transform=transforms.Compose([Normalizer(),
                                                          Augmenter(),
                                                          Resizer()]))

    ddsampler = torch.utils.data.distributed.DistributedSampler(dataset)
    loader = DataLoader(dataset, batch_size=config.batch_size, sampler=ddsampler, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=resnet50, config=config, pretrained=True)
    model = training.eval.model_load_gen(model, file, parallel_trained=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank)

    run.training(ddp_model, loader, logname="resnet50_cosR",starting_epoch=startepoch,ending_epoch=endepoch)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["CUDA_VISIBLE_DEVICES"] = "1,0"
    world_size = 2
    #mp.spawn(training_process_checkpoint,args=(world_size, cfg), nprocs=world_size)
    mp.spawn(training_process,args=(world_size, cfg), nprocs=world_size)