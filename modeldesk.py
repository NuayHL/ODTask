import torch
import numpy as np
from training.assign import AnchAssign

gt1 = np.array([[34,45,78,656],[100,80,200,234]],dtype=np.float32)
gt = [gt1,gt1]
test = AnchAssign()
output = test.assign(gt)

