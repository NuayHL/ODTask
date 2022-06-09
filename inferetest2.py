#各种各样的测试都在这里
import os
print(os.getcwd())

'''
{"imgs":List lenth B, each with np.float32 img
"anns":List lenth B, each with np.float32 ann}
'''



from training.loss import Defaultloss
from models.yolo import YOLOv3
from models.resnet import resnet50
from training.assign import AnchAssign
from training.config import cfg
from training.running import model_load_gen
from util.visualization import show_bbox
from data.eval import inference_single_visualization

model = YOLOv3(numofclasses=1,backbone=resnet50)
model = model_load_gen(model, "70E_8B_800_1024_resnet50_4nd_n_E60",parallel_trained=False)
model = model.to(cfg.pre_device)

img = 'IMG_20220609_141503.jpg'

inference_single_visualization(img, model)



