from util.visualization import dataset_inspection
from data.dataset import Txtdatset

datset = Txtdatset("CrowdHuman/Images_val","CrowdHuman/label/Images_val", xywhoutput=True)
dataset_inspection(datset, 800, anntype='xywh')