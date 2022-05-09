import cv2

def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')

def show_bbox(img, bboxs=[], type="default",color=[0,0,255],**kwargs):
    '''
    :param img: str for file path/np.ndarray (w,h,c)
    :param bboxs: one or list
    :param type: bbox format
    :param color: red
    :param kwargs: related to cv2.rectangle
    :return:
    '''
    assert type in ["default","diagonal","crowdhuman"],"the bbox format should be \'default\' or \'diagonal\' or \'crowdhuman\'"
    if isinstance(img, str): img = cv2.imread(img)
    bboxs = bboxs if _isArrayLike(bboxs) else [bboxs]
    for bbox in bboxs:
        bbox[0] = int(bbox[0])
        bbox[1] = int(bbox[1])
        bbox[2] = int(bbox[2])
        bbox[3] = int(bbox[3])
        if type == "diagonal": a, b = (bbox[0],bbox[1]),(bbox[2],bbox[3])
        elif type == "crowdhuman": a, b = (bbox[0],bbox[1]),(bbox[0]+bbox[2],bbox[1]+bbox[3])
        else: a, b = (bbox[0]-int(bbox[2]/2),bbox[1]-int(bbox[3]/2)),(bbox[0]+int(bbox[2]/2),bbox[1]+int(bbox[3]/2))
        cv2.rectangle(img,a,b,color)
    cv2.imshow("",img)
    cv2.waitKey()
    cv2.destroyAllWindows()

