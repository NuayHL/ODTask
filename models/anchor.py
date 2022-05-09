import numpy as np
from training.config import cfg

def generateAnchors(fpnlevels=None, basesize=None, ratios=None, scales=None):
    if fpnlevels == None:
        fpnlevels = [3,4,5]
    if basesize == None:
        basesize = [2**(x+2) for x in fpnlevels]
    if ratios == None:
        # ratios = h/w
        ratios = cfg.anchorRatio
    if scales == None:
        scales = cfg.anchorScales

    # formate = x1y1x2y2
    allAnchors = np.zeros((0,4)).astype(np.float32)
    for idx, p in enumerate(fpnlevels):
        # need float(stride[0]/2)!
        stride = [cfg.input_width/2**p, cfg.input_height/2**p]
        xgrid = np.arange(0,cfg.input_width,stride[0]) + stride[0]/2
        ygrid = np.arange(0,cfg.input_height,stride[1]) + stride[1]/2
        xgrid, ygrid = np.meshgrid(xgrid, ygrid)






