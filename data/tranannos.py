import json
from training.config import cfg
from util.primary import progressbar

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

if __name__ == "__main__":
    modify_area("CrowdHuman/annotation_val_coco_style.json")