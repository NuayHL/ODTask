import json
import cv2
from util import progressbar

def txt2coco(filepaths: list, outputname):
    '''
    :param filepaths: pathfile list
    :param outputname: the name of output json file
    :return: None

    About the WiderPerson Dataset:
    CrowdHuman contains 8000 training imgs,
                        1000 validation imgs,
                        4382 testing imgs.

    About COCO format:{
        "info":{"year"}
        "images":[image]
        "annotations":[annotation]
        "categories":{}
    }
    image{
        "id": int,
        "width": int,
        "height": int,
        "file_name": str
    }
    annotation{
        "id": int,
        "image_id": int,
        "category_id": int,
        "bbox": [x,y,width,height] (fbox),
        "iscrowd": 0 or 1
    }
    categories[{
        "id": int,
        "name": str,
        "supercategory": str,
    }]
    '''
    info = dict()
    images = []
    annotations = []
    categories = json.load(open("data/categories_coco.json"))

    info["year"] = 2019
    print("begin convert %s to coco format"%filepaths)

    imageinfo = []
    for path in filepaths:
        with open(path) as f:
            imageinfo += f.readlines()

    bbox_id = 0
    num_imgs = len(imageinfo)
    for idx, imgname in enumerate(imageinfo):
        id = idx + 1
        if imgname[-1] == "\n":
            imgname = imgname[:-1]
        h, w, _ = (cv2.imread("WiderPerson/Images/"+imgname+".jpg")).shape
        img_info = {"id":id, "width":w, "height":h, "file_name":imgname}
        images.append(img_info)

        with open("WiderPerson/Annotations/"+imgname+".jpg.txt","r") as f:
            bboxs = f.readlines()
        for bbox in bboxs[1:]:
            bbox = [int(num) for num in bbox.split()]
            if bbox[0] >= 4 : continue
            bbox_id += 1
            bbox[3] -= bbox[1]
            bbox[4] -= bbox[2]
            area = bbox[3] * bbox[4] #fbox area
            bbox_info={"id":bbox_id,"image_id":id,"category_id":1,
                       "bbox":bbox[1:], "area":area, "iscrowd":0}
            annotations.append(bbox_info)
        progressbar(float(id/num_imgs))

    output = {"info":info, "images":images, "annotations":annotations, "categories":categories}
    json.dump(output, open("WiderPerson/"+outputname+".json",'w'))

if __name__ == '__main__':
    txt2coco(["WiderPerson/train.txt"],"widerperson_train_coco_style")
    txt2coco(["WiderPerson/val.txt"], "widerperson_val_coco_style")
    txt2coco(["WiderPerson/train.txt","WiderPerson/val.txt"],"widerperson_all_coco_style")

