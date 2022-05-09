import json

def cal_img_size(coco_style_annotation_path):
    with open(coco_style_annotation_path) as f:
        annotation = json.load(f)
    assert "images" in annotation.keys(),"Unknown format of annotation. The annotation must have key:\'images\'"
    width = 0.
    height = 0.
    imgnum = len(annotation["images"])
    for imgs in annotation["images"]:
        width += imgs["width"]
        height += imgs["height"]
    return width/imgnum, height/imgnum

if __name__ == "__main__":
    print(cal_img_size("../CrowdHuman/annotation_train_coco_style.json"))
    # (965.7074, 1355.6417333333334)