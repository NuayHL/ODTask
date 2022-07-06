import torch

import numpy as np
from torch import Tensor
from torchvision.ops import batched_nms
from training.iou import IOU

class NMS():
    def __init__(self, nmstype="nms"):
        self.nmstype = nmstype
    def __call__(self, inputtensor, threshold):
        """
        :param inputtensor: N x (x1y1x2y2 score category)
        :return:
        """
        if self.nmstype == "nms":
            return self._nms(inputtensor, threshold)
        elif self.nmstype == "softnms":
            return self._softnms(inputtensor)

    def _nms(self, input, threshold):
        return batched_nms(input[:,:4],input[:,4],input[:,5],threshold)

    def _softnms(self, input: Tensor, threshold):
        iou = IOU()
        result = []
        idxs = input[:, 4].argsort()
        while len(idxs) != 0:
            if len(idxs) == 1:
                result.append(idxs[-1])
                break
            maxone = idxs[-1]
            iou_with_rest = iou(torch.unsqueeze(input[:4, maxone], 0), input[:4, idxs[:-1]])

    def change(self):
        self.nmstype = 1

if __name__ == "__main__":
    nms = NMS("softnms")



