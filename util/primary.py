import matplotlib.pyplot as plt
from torch import nn
from training.config import Config
#小工具

def printImg(img, title: str='', type = 0):
    if type == 0: plt.imshow(img)
    else: plt.imshow(img, cmap=plt.cm.gray)
    plt.title(title)
    plt.axis('off')
    plt.show()

def progressbar(percentage, endstr='', barlenth=20):
    if int(percentage)==1: endstr +='\n'
    print('\r[' + '>' * int(percentage * barlenth) +
          '-' * (barlenth - int(percentage * barlenth)) + ']',
          format(percentage * 100, '.1f'), '%', end=' '+endstr)

def numofParameters(model: nn.Module ):
    nump = 0
    for par in model.parameters():
        nump += par.numel()
    return nump

def cfgtoStr(cfg:Config):
    name = str(cfg.trainingEpoch)+"E_"+str(cfg.batch_size)+"B_"+str(cfg.input_width)+ \
           "*"+str(cfg.input_height)
    return name






