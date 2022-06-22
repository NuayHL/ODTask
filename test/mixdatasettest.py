import numpy as np
from data.dataset import CocoDataset, MixCocoDatset
from util.visualization import dataset_inspection

widerdataset = CocoDataset("WiderPerson/widerperson_train_coco_style.json",
                      "WiderPerson/Images",transform=None, bbox_type="bbox")

crowddataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json",
                          "CrowdHuman/Images_train")

widerdataset_val = CocoDataset("WiderPerson/widerperson_val_coco_style.json",
                      "WiderPerson/Images",transform=None, bbox_type="bbox")
maindatset = MixCocoDatset([crowddataset,widerdataset],None)

print(len(maindatset))
maindatset.addDataset(widerdataset_val)
print(len(maindatset))


dataset_inspection(maindatset, 14999)


