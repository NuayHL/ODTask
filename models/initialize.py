import torch
import torch.nn as nn
import math
import numpy as np

def weight_init(m):
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def seed_init(num=None):
    if num == None: return 0
    np.random.seed(num)
    torch.manual_seed(num)
    torch.cuda.manual_seed_all(num)