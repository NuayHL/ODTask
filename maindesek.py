import scipy.io as scio
import numpy as np

annots = scio.loadmat("CityPersons/annotations/anno_train.mat")
annos = annots["anno_train_aligned"]
print(annos[0][101])