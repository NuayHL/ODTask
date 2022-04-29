#各种各样的测试都在这里

from util.visualization import show_bbox
from data.trandata import CrowdHDataset
from torch.utils.data import DataLoader

ID = 56

dataset = CrowdHDataset("CrowdHuman/annotation_train_coco_style.json")
mix = dataset[ID]

show_bbox(mix["img"], mix["anns"],type = "crowdhuman")






