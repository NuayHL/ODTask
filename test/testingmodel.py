from data.dataset import CocoDataset, Normalizer, Resizer
from models.yolo import YOLOv3
from torchvision.transforms import Compose
from training.eval import model_eval_loss



if __name__ == '__main__':
    testset = CocoDataset('CrowdHuman/annotation_val_vbox_coco_style.json', "CrowdHuman/Images_val",
                          transform=Compose([Normalizer(), Resizer()]))
    #loss: 0.9178592605517362

    #trainset = CocoDataset('CrowdHuman/annotation_train_coco_style.json', type='train',
                            #transform=Compose([Normalizer(), Resizer()]))
    #loss: 0.21466813359703538

    model = YOLOv3(numofclasses=1, istrainig=True, backbone=None)

    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E60", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E65", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E70", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E75", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E80", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E85", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E90", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E95", testset)
    model_eval_loss(model, "70E_8B_800_1024_darknet53_from55_E100", testset)
