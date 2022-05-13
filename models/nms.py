import torch
import numpy as np

class NMS():
    def __init__(self, nmstype="nms"):
        self.nmstype = nmstype
    def __call__(self, inputtensor):
        if self.nmstype == "nms":
            return self._nms(inputtensor)
    def _nms(self, inputtensor):
        pass