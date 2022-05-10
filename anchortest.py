import os
print(os.getcwd())
from models.anchor import generateAnchors
import numpy as np
import torch


anchors = generateAnchors()
print(anchors)
print(anchors.shape)