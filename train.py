import argparse
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP


parser = argparse.ArgumentParser()
parser.add_argument()


def training_process(rank, world_rank):
    pass

if __name__ == "__main__":
    mp.spawn()