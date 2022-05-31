import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from .config import cfg
from util.mylogger import mylogger
from util.primary import cfgtoStr
import torch.optim as optim
import torch.optim.lr_scheduler as sche
from torch.distributed import is_initialized, get_rank
from util.primary import DDPsavetoNormal

def model_save_gen(model:nn.Module, filename, path="models/model_pth"):
    torch.save({"GEN":model.state_dict()}, path+"/"+filename+".pt")

def model_load_gen(model:nn.Module, filename, path="models/model_pth", parallel_trained=False):
    state_dict = torch.load(path+"/"+filename+".pt")
    if parallel_trained:
        state_dict = DDPsavetoNormal(state_dict["GEN"])
    else:
        state_dict = state_dict["GEN"]
    model.load_state_dict(state_dict)
    return model

def training(model:nn.Module, loader:DataLoader, optimizer=None, scheduler=None, logname=None, _cfg=cfg, save_per_epoch=5, **kwargs):
    '''
    :param model: model for training
    :param loader: dataloader for training
    :param optimizer: optimizer, if not specific, using Adam with lr=0.0005
    :param scheduler: scheduler, if not specific, using MultiStepLR with [20, 50, 60], 0.1
    :param logname: the diy name adding to the cfg_formate
    :param _cfg: cfg
    :param save_per_epoch: save the model state_dict using [final name] per specifc epoch
    :param kwargs: using for the optimizer parameters
    :return:
    '''
    if optimizer==None:
        optimizer = optim.Adam(model.parameters(),lr=0.001)
    else:
        optimizer = optimizer(model.parameters(), **kwargs)
    if scheduler==None:
        scheduler = sche.MultiStepLR(optimizer, milestones=[20,50,60], gamma=0.1)

    # initialize rank
    if is_initialized(): rank = get_rank()
    else: rank = _cfg.pre_device

    # initialize logname
    if logname is None:
        name = cfgtoStr(_cfg)
    else: name = cfgtoStr(_cfg) + "_" + logname

    if is_initialized():
        name += ("_gpu"+str(rank))

    # initialize logger
    logger = mylogger(name,rank)

    # begin training
    model.train()
    lenepoch = len(loader)

    for i in range(_cfg.trainingEpoch):
        if is_initialized():
            loader.sampler.set_epoch(i)
        for idx, batch in enumerate(loader):
            optimizer.zero_grad()
            batch["imgs"] = batch["imgs"].to(rank)
            loss = model(batch)
            loss.backward()
            optimizer.step()
            logger.info("epoch "+str(i+1)+"/"+str(_cfg.trainingEpoch)+":"+str(idx+1)+"/"+str(lenepoch)
                        +"//loss:"+str(loss.item()))
        scheduler.step()
        if (i+1)%save_per_epoch == 0:
            logger.warning("Saving Models!")
            if rank == 0 or not is_initialized():
                model_save_gen(model,name+":E"+str(i+1))


