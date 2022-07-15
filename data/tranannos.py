import json
from training.config import cfg
from util.primary import progressbar
from pycocotools.coco import COCO

def modify_area(coco_style_json, width=cfg.input_width, height=cfg.input_height):
    with open(coco_style_json,"r") as f:
        all_ann = json.load(f)
    lenth = len(all_ann["annotations"])
    for ann in all_ann["annotations"]:
        ori_img = all_ann["images"][ann["image_id"]-1]
        assert ori_img["id"] == ann["image_id"]
        ori_w, ori_h = ori_img["width"], ori_img["height"]
        ann["area"] *= (width/float(ori_w) * height/float(ori_h))
        progressbar(ann["id"]/lenth)
    with open(coco_style_json[:-5]+"_%s_%s.json"%(str(width),str(height)),"a") as f:
        json.dump(all_ann, f)

def coco2txt(coco_style_json: str, outputpath: str):
    labels = COCO(coco_style_json)
    imgs = labels.getImgIds()
    lenth = len(imgs)
    for idx, imgid in enumerate(imgs):
        img = labels.loadImgs(imgid)[0]
        w, h = img["width"], img["height"]
        anns_idx = labels.getAnnIds(imgid)
        anns = labels.loadAnns(anns_idx)
        with open(outputpath + "/" + img["file_name"]+".txt", "a") as f:
            for ann in anns:
                if ann["category_id"] == 0:
                    continue
                ann["bbox"][0] = ann["bbox"][0] if ann["bbox"][0]>=0 else 0.
                ann["bbox"][1] = ann["bbox"][1] if ann["bbox"][1]>=0 else 0.
                ann["bbox"][2] = ann["bbox"][2] if ann["bbox"][0] + ann["bbox"][2] <= w \
                                else w - ann["bbox"][0]
                ann["bbox"][3] = ann["bbox"][3] if ann["bbox"][1] + ann["bbox"][3] <= h \
                                else h - ann["bbox"][1]

                bboxsys = str(ann["category_id"]-1) + " "
                bboxsys += str((ann["bbox"][0] + ann["bbox"][2] / 2) / w) + " "
                bboxsys += str((ann["bbox"][1] + ann["bbox"][3] / 2) / h) + " "
                bboxsys += str(ann["bbox"][2] / w) + " "
                bboxsys += str(ann["bbox"][3] / h) + "\n"
                f.write(bboxsys)
        progressbar((idx+1)/float(lenth), barlenth = 40)

def coco2yolov6result(coco_style_json: str):
    with open(coco_style_json,"r") as f:
        all_ann = json.load(f)
    id_filename = {}
    for ann in all_ann["images"]:
        id_filename[ann["id"]] = ann["file_name"]
        ann["id"] = ann["file_name"]
    for ann in all_ann["annotations"]:
        ann["image_id"] = id_filename[ann["image_id"]]
        ann["height"] = ann["bbox"][3]
    with open(coco_style_json[:-5]+"_filename_id.json","a") as f:
        json.dump(all_ann, f)

if __name__ == "__main__":
    # coco2txt("CrowdHuman/annotation_train_coco_style.json", "labels/Images_train")
    # coco2txt("CrowdHuman/annotation_val_coco_style.json", "labels/Images_val")
    coco2yolov6result("CrowdHuman/annotation_val_coco_style.json")