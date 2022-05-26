import numpy as np
import cv2

class Results():
    def __init__(self, bboxes, classes, scores):
        self.bboxes = bboxes
        self.classes = classes
        self.scores = scores

    def load_bboxex(self):
        return self._xywh_to_x1y1x2y2(self.bboxes)


    def _xywh_to_x1y1x2y2(self,input):
        input[:, 0] = input[:,0] - 0.5 * input[:,2]
        input[:, 1] = input[:,1] - 0.5 * input[:,3]
        input[:, 2] = input[:,0] + input[:,2]
        input[:, 3] = input[:,1] + input[:,3]



def average_precision():
    pass

def average_recall():
    pass






