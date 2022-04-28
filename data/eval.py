import numpy as np
import cv2

class Results():
    def __init__(self, id, results):
        self.id = id
        assert isinstance(results,list) and len(results[0])==6,\
            'Invalid results format. results format should be: [[x,y,w,h,confidence,class],]'
        self.results = results



def average_precision():
    pass

def average_recall():
    pass






