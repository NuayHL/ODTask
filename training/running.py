import torch.nn as nn
from torch.utils.data import DataLoader
from .config import cfg
from util.mylogger import mylogger
from util.primary import cfgtoStr
from training.eval import coco_eval, model_save_gen
import torch.optim as optim
import torch.optim.lr_scheduler as sche
from torch.distributed import is_initialized, get_rank

def training(model:nn.Module, loader:DataLoader, optimizer=None, scheduler='steplr', valdataset=None,
             logname=None, _cfg=cfg, save_per_epoch=5, starting_epoch=0,ending_epoch=None, **kwargs):
    '''
    :param model: model for training
    :param loader: dataloader for training
    :param optimizer: optimizer, if not specific, using Adam with lr=0.0005
    :param scheduler: scheduler, if not specific, using MultiStepLR with [20, 50, 60], 0.1
    :param logname: the diy name adding to the cfg_formate
    :param _cfg: cfg
    :param save_per_epoch: save the model state_dict using [final name] per specifc epoch
    :param starting_epoch: training from the checkpoint
    :param ending_epoch: specific ending epoch for training, usually smaller than cfg.epoch.
    :param kwargs: using for the optimizer parameters
    :return:
    '''
    if optimizer==None:
        optimizer = optim.Adam(model.parameters(),lr=0.0015)
    else:
        optimizer = optimizer(model.parameters(), **kwargs)
    if scheduler is None:
        scheduler = 'steplr'
    if scheduler=='steplr':
        scheduler = sche.MultiStepLR(optimizer, milestones=[85, 95], gamma=0.1)
    # elif scheduler=='cosineRestarts':
    #     scheduler = CosineAnnealingWarmupRestarts(optimizer, first_cycle_steps=20, max_lr=0.1, min_lr=0.0001, warmup_steps=5, gamma=0.8 )
    else:
        raise NotImplementedError('invalid scheduler')

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
    logger = mylogger(name,rank, is_initialized())

    # handle check point problem
    if ending_epoch is None:
        ending_epoch = _cfg.trainingEpoch
    assert starting_epoch < ending_epoch,'starting_epoch should smaller or equal to ending_epoch'

    # if load from checking points
    for i in range(starting_epoch):
        scheduler.step()

    # begin training
    lenepoch = len(loader)
    for i in range(starting_epoch, ending_epoch):
        model.istraining = True
        model.train()
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
        current_state = name+"_E"+str(i+1)

        # Saving models or Evaluating when certain Epoch
        if i+1 == ending_epoch:
            if ending_epoch==_cfg.trainingEpoch:
                logger.warning("Congratulations! Training complete. Saving Models!")
            else:
                logger.warning("Reaching checkpoint. Saving Models!")
        elif (i+1)%save_per_epoch == 0:
            logger.warning("Saving Models!")
        else:
            continue

        if rank == 0 or not is_initialized():
            model_save_gen(model, current_state)
            if valdataset != None:
                logger.warning("Begin Evaluating...")
                model.istraining = False
                model.eval()
                coco_eval(model, valdataset,
                          result_name=current_state, logname=name + "_eval")
                logger.warning("Evaluating Complete!")

