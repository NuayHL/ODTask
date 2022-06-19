#各种各样的测试都在这里
import os
print(os.getcwd())
os.environ["CUDA_VISIBLE_DEVICES"]  = '0,1'

'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann}
'''

from data.trandata import CocoDataset

from models.yolo import YOLOv3
from models.resnet import resnet50
from training.config import cfg
from training.eval import inference_dataset_visualization, model_load_gen

id = 2555

dataset = CocoDataset("CrowdHuman/annotation_train_coco_style.json","CrowdHuman/Images_train")
#dataset = CocoDataset("CrowdHuman/annotation_val_vbox_coco_style.json")

model = YOLOv3(numofclasses=1,backbone=resnet50)
model = model_load_gen(model, "70E_8B_800_1024_resnet50_4nd_E60",parallel_trained=False)
model = model.to(cfg.pre_device)

inference_dataset_visualization(dataset, id, model)


