import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import cfg
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel as DDP

def model_save_gen(model:nn.Module, filename, path="models/yolov3_pth"):
    torch.save({"GEN":model.state_dict()}, path+"/"+filename+".pt")

def model_load_gen(model:nn.Module, filename, path="models/yolov3_pth"):
    state_dict = torch.load(path+"/"+filename+".pt")
    model.load_state_dict(state_dict["GEN"])
    return model

def training(model:nn.Module, loader:DataLoader, optimizer=None, epoch=cfg.trainingEpoch):
    if optimizer==None:
        optimizer = optim.Adam(model.parameters())

    model.train()
    lenepoch = len(loader)

    for i in range(epoch):
        for idx, batch in enumerate(loader):
            batch["imgs"] = batch["imgs"].to(cfg.pre_device)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            print("epoch",i,"/",cfg.trainingEpoch,":",idx,"/",lenepoch,"//loss:",loss.item())


