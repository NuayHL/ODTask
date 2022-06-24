import scipy.io as scio
import numpy as np

from data.dataset import CocoDataset
from util.visualization import dataset_inspection

dataset2 = CocoDataset("WiderPerson/widerperson_all_coco_style.json",
                       "WiderPerson/Images", bbox_type="bbox",transform=None)

id = 4573

sampel = dataset2[id]

print(sampel["anns"])

dataset_inspection(dataset2, id)