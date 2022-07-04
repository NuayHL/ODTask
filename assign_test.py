from util.visualization import assign_hot_map, dataset_inspection
from training.assign import AnchAssign
from models.anchor import generateAnchors
from data.dataset import CocoDataset, Resizer
from torchvision import transforms

id = 7000

dataset1 = CocoDataset("CrowdHuman/annotation_train_coco_style.json","CrowdHuman/Images_train",
                      bbox_type="bbox",transform=transforms.Compose([Resizer()]),ignored_input=True)

dataset2 = CocoDataset("CrowdHuman/annotation_train_coco_style.json","CrowdHuman/Images_train",
                      bbox_type="bbox",transform=transforms.Compose([Resizer()]),ignored_input=False)

dataset_inspection(dataset1,id)

gts1 = dataset1[id]["anns"]
gts2 = dataset2[id]["anns"]
anchors = generateAnchors(fpnlevels=[3,4,5,6,7], singleBatch=True)
assign_method1 = AnchAssign(anchors, using_ignored_input=True)
assign_method2 = AnchAssign(anchors, using_ignored_input=False)
assign_result1 = assign_method1.assign([gts1])[0]
assign_result2 = assign_method2.assign([gts2])[0]
print(assign_result1.shape)
print((assign_result2-assign_result1).sum())

assign_hot_map(anchors, assign_result1, dataset1[id]["img"], gt=gts1)
assign_hot_map(anchors, assign_result2, dataset1[id]["img"], gt=gts2)