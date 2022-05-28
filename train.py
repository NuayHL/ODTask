import os
import torch
import training.running as run
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from models.yolo import YOLOv3

from data.trandata import CrowdHDataset, OD_default_collater
from torch.utils.data import DataLoader
from training.config import cfg


def training_process(rank, world_size):
    dist.init_process_group("nccl",rank=rank, world_size=world_size)
    dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
    ddsampler = torch.utils.data.distributed.DistributedSampler(dataset)

    loader = DataLoader(dataset, batch_size=cfg.batch_size, sampler=ddsampler, collate_fn=OD_default_collater)

    model = YOLOv3(numofclasses=1, istrainig=True)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model).to(rank)
    ddp_model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    run.training(ddp_model, loader)

if __name__ == "__main__":
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    world_size = 2
    mp.spawn(training_process,args=(world_size,), nprocs=world_size)
