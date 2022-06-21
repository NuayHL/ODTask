import torch
import numpy as np
from torchvision.ops import batched_nms

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
            return self._softnms()

    def _nms(self, input, threshold):
        return batched_nms(input[:,:4],input[:,4],input[:,5],threshold)

    def _softnms(self):
        pass