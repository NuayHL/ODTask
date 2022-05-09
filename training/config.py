# generate cfg for training from yaml
import yaml

class Config():
    def __init__(self,cfgPath):
        with open(cfgPath,"r") as f:
            self.cfg = yaml.safe_load(f.read())
        #input
        self.input_width = self.cfg["input"]["width"]
        self.input_height = self.cfg["input"]["height"]
        self.input_bboxtype = self.cfg["input"]["bboxtype"]

        #assign_cfg
        self.assignType = self.cfg["assign_cfg"]["assignType"]
        self.ioutype = self.cfg["assign_cfg"]["ioutype"]
        self.threshold = self.cfg["assign_cfg"]["threshold"]

cfg = Config("training/config.yaml")

