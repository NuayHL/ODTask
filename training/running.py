import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import cfg
from util.mylogger import mylogger
from util.primary import cfgtoStr
import torch.optim as optim
import torch.optim.lr_scheduler as sche
from torch.distributed import is_initialized, get_rank

def model_save_gen(model:nn.Module, filename, path="models/yolov3_pth"):
    torch.save({"GEN":model.state_dict()}, path+"/"+filename+".pt")

def model_load_gen(model:nn.Module, filename, path="models/yolov3_pth"):
    state_dict = torch.load(path+"/"+filename+".pt")
    model.load_state_dict(state_dict["GEN"])
    return model

def training(model:nn.Module, loader:DataLoader, optimizer=None, scheduler=None, _cfg=cfg):
    if optimizer==None:
        optimizer = optim.Adam(model.parameters())
    if scheduler==None:
        scheduler = sche.MultiStepLR(optimizer, milestones=[20,50,60], gamma=0.1)

    name = cfgtoStr(_cfg)
    if is_initialized():
        name += ("_gpu"+str(get_rank()))

    logger = mylogger(name,get_rank())

    model.train()
    lenepoch = len(loader)

    for i in range(_cfg.trainingEpoch):
        if is_initialized():
            loader.sampler.set_epoch(i)
        for idx, batch in enumerate(loader):
            batch["imgs"] = batch["imgs"].to(_cfg.pre_device)
            optimizer.zero_grad()
            loss = model(batch)
            loss.backward()
            optimizer.step()
            logger.info("epoch"+str(i)+"/"+str(_cfg.trainingEpoch)+":"+str(idx)+"/"+str(lenepoch)
                        +"//loss:"+str(loss.item()))
        scheduler.step()
        if (i+1)%5 == 0:
            logger.warning("Saving Models!")
            if get_rank() == 0:
                model_save_gen(model,"70E_4B_800*1024:E"+str(i+1))


