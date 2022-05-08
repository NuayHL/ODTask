# generate cfg for training from yaml
import yaml

class Config():
    def __init__(self,cfgPath):
        with open(cfgPath,"r") as f:
            self.cfg = yaml.safe_load(f.read())
        self.assignType = self.cfg["assign_cfg"]["assignType"]
        self.ioutype = self.cfg["assign_cfg"]["ioutype"]
        self.threshold = self.cfg["assign_cfg"]["threshold"]
    def __call__(self):
        return self.cfg


config = Config("config.yaml")
cfg = config()
