# Some basic setup:
# Setup detectron2 logger

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import os, json, cv2, random

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

import os
module_path = os.path.abspath(__file__)
dir_path = os.path.dirname(module_path)
# import torch.multiprocessing as mp
# mp.set_start_method('spawn')

def faster_rcnn_detection(image_name):
    # cftracker\detectron2_detector\person23_000292.jpg
    image_path = os.path.join(dir_path,"to_detect/"+image_name)
    print("image_path",image_path)
    im = cv2.imread(image_path)

    # Inference with a keypoint detection model
    cfg = get_cfg()   # get a fresh new config
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7  # set threshold for this model

    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(im)
    # print(outputs['instances'].get("pred_boxes").tensor.cpu().numpy().tolist()==[])
    if(outputs['instances'].get("pred_boxes").tensor.cpu().numpy().tolist()==[]):
        return None
    else:
        center = outputs['instances'].get("pred_boxes").get_centers().cpu().numpy()[0].tolist()
        return center[0],center[1]
    # print("outputs['instances'].get('pred_boxes').nonempty().cpu()")
    # print(outputs['instances'].get("pred_boxes").nonempty().cpu())
    # print("outputs['instances'].get('pred_boxes').nonempty().cpu()")
    # print("outputs['instances'].get('pred_boxes').tensor.cpu().numpy()",outputs['instances'].get('pred_boxes').tensor.cpu().numpy())
    #Boxes(tensor([], device='cuda:0', size=(0, 4)))


if __name__ == "__main__":
    image_name = "person8_000935.jpg"
    print(faster_rcnn_detection(image_name))
    # print(float("hello"))
