import json



if __name__ == '__main__':
    f = open('../CrowdHuman/annotation_train.odgt')
    for imggt in f:
        imggt = json.loads(imggt)
        create = open('../CrowdHuman/annotation_train/'+imggt['ID']+'.txt','w')
        for objects in imggt['gtboxes']:
            m = objects['tag']+' '
            print(m+str(objects['vbox']),file=create)
        create.close()
    f.close()
