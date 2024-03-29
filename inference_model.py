import matplotlib.pyplot as plt
import numpy as np

from models.yolo import YOLOv3
from training.eval import model_inference_coconp, checkpoint_load, coco_eval
from data.dataset import CocoDataset, Resizer, Normalizer
from torchvision.transforms import Compose

dataset = CocoDataset("CrowdHuman/annotation_val_coco_style_1024_800.json", "CrowdHuman/Images_val",
                      transform=Compose([Normalizer(), Resizer()]))

# model = YOLOv3(numofclasses=1, backbone=None, istrainig=False)
# model = model_load_gen(model, "70E_8B_800_1024_darknet53_from55_E75",parallel_trained=False)
# model = model.to(0)
#
# result = model_inference_coconp(dataset, model)
#
# np.save('CrowdHuman/70E_8B_800_1024_Darknet53_E75_0.7.npy', result)

result = np.load('CrowdHuman/120E_8B_800_1024_yolo_resnet101_1st_E5.npy')

print(result.shape)

coco_eval(None, dataset, logname="debugtest",resultnp=result)



# index = np.arange(5,101,5)
# loss = [1.6161321055601565,
#         1.021973163342578,
#         0.8153387874900778,
#         0.7021312630958045,
#         0.6985450383618342,
#         0.6242788860517096,
#         0.6011593557869583,
#         0.6101775371862963,
#         0.6214486029582925,
#         0.6381067989107574,
#         0.6754159038797549,
#         0.7503992788616757,
#         0.6759851359860121,
#         0.9211605182314745,
#         0.8149113345899416,
#         0.8451984832391309,
#         0.9038754368079559,
#         1.1967515699801772,
#         1.4616554965221011,
#         1.500012654081472]
#
# fig, ax = plt.subplots()
# ax.plot(index, loss)
# ax.set(xlabel="Epochs", ylabel="Loss", title="haha")
# ax.grid()
# plt.show()