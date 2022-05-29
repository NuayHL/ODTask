import cv2
import matplotlib.pyplot as plt

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

def draw_loss(file_path,name="loss"):
    with open(file_path,"r") as f:
        losses = f.readlines()
        loss_list = []
        index = []
        start_idx = 0
        for i in losses:
            if "WARNING" in i:
                continue
            loss = float(i[(i.rfind(":")+1):])
            loss_list.append(loss)
            index.append(start_idx)
            start_idx += 1
    fig, ax = plt.subplots()
    ax.plot(index, loss_list)
    ax.set(xlabel="Iteration(times)",ylabel="Loss",title="Training Loss for "+file_path)
    ax.grid()

    fig.savefig(name+".png")
    plt.show()
