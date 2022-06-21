# generate cfg for training from yaml
import yaml

class Config():
    def __init__(self,cfgPath):
        with open(cfgPath,"r") as f:
            self.cfg = yaml.safe_load(f.read())
        #device
        self.pre_device = self._device_parse(self.cfg["device"]["preprocess"])

        #input
        self.input_width = self.cfg["input"]["width"]
        self.input_height = self.cfg["input"]["height"]
        self.input_bboxtype = self.cfg["input"]["bboxtype"]
        self.batch_size = self.cfg["input"]["batchSize"]

        #assign cfg
        self.assignType = self.cfg["assign_cfg"]["assignType"]
        self.assignIouType = self.cfg["assign_cfg"]["ioutype"]
        self.assign_threshold = self.cfg["assign_cfg"]["threshold"]

        #anchor settings
        self.anchorLevels = self.cfg["anchors"]["fpnlevels"]
        self.anchorRatio = self.cfg["anchors"]["ratios"]
        self.anchorScales = self.cfg["anchors"]["scales"]

        #training
        self.trainingEpoch = self.cfg["training"]["epoch"]

        #inference
        self.background_threshold = self.cfg["inference"]["background_threshold"]
        self.class_threshold = self.cfg["inference"]["class_threshold"]
        self.nms_threshold = self.cfg["inference"]["nms_threshold"]

    def _device_parse(self,str):
        if str == "cuda1":
            return "cuda:1"
        elif str == "cuda0":
            return "cuda:0"
        elif str == "cuda":
            return str
        else:
            return "cpu"

cfg = Config("training/cfg/config.yaml")

