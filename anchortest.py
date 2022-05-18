import torch
from models.anchor import generateAnchors
from training.assign import AnchAssign
a = generateAnchors()
print(a.shape)
b = AnchAssign()
result = b.assign()

