import numpy as np
from training.config import cfg

def generateAnchors(fpnlevels=None, basesize=None, ratios=None, scales=None, singleBatch=False):
    '''
    return: batch_size X total_anchor_numbers X 4
    singleBatch: set True if only need batchsize at 1
    '''
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
        stride = [2**p, 2**p]
        xgrid = np.arange(0,cfg.input_width,stride[0]) + stride[0]/2.0
        ygrid = np.arange(0,cfg.input_height,stride[1]) + stride[1]/2.0
        xgrid, ygrid = np.meshgrid(xgrid, ygrid)
        anchors = np.vstack((xgrid.ravel(),ygrid.ravel()))
        lenAnchors = anchors.shape[1]
        anchors = np.tile(anchors,(2,len(ratios)*len(scales))).T
        start = 0
        for ratio in ratios:
            for scale in scales:
                anchors[start:start+lenAnchors, 0] -= basesize[idx] * scale / 2.0
                anchors[start:start+lenAnchors, 1] -= basesize[idx] * scale * ratio / 2.0
                anchors[start:start+lenAnchors, 2] += basesize[idx] * scale / 2.0
                anchors[start:start+lenAnchors, 3] += basesize[idx] * scale * ratio / 2.0
                start += lenAnchors
        allAnchors = np.append(allAnchors,anchors,axis=0)

    if singleBatch: return allAnchors

    allAnchors = np.tile(allAnchors, (cfg.batch_size,1,1))

    return allAnchors






