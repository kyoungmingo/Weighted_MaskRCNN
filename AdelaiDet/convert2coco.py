
###########################################################################################
import os
import numpy as np
import json
from detectron2.structures import BoxMode
import xlrd
import math
import random

directory = "/mnt/ssm/elipse/sample1"

def get_microcontroller_dicts(directory):
    classes = ['Microcystis']
    dataset_dicts = []
    scaler = np.arange(0.00000000000001,0.00000000000002,0.00000000000000001)
    for filename in [file for file in os.listdir(directory) if file.endswith('.xlsx')]:
        xlsx_file = os.path.join(directory, filename)
        wb = xlrd.open_workbook(xlsx_file)
        record = {}
        imagename = os.path.join(directory,filename.split('.')[0] + ".png")
        record["file_name"] = imagename
        record["height"] = 512
        record["width"] = 512
        objs = []

        for j in np.arange(len(wb.sheet_names())):
            ws = wb.sheet_by_index(j)
            px = []
            py = []
            px1 = []
            py1 = []

            for i in np.arange(ws.nrows):
                # if i == math.floor((ws.nrows)/2) - 1:
                if i == ws.nrows - 1:
                    break
                else:
                    # py.append(ws.col_values(1)[2*i+1])
                    # px.append(ws.col_values(2)[2*i+1])
                    py.append(ws.col_values(1)[i+1])
                    px.append(ws.col_values(2)[i+1])
            px1=random.sample(list(scaler),len(px))
            py1 =random.sample(list(scaler),len(py))
            px = list(np.array(px)-np.array(px1))
            py = list(np.array(py)-np.array(py1))
            poly1 = [(x, y) for x, y in zip(px, py)]
            poly1 = [p for x in poly1 for p in x]
            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly1],
                "category_id": classes.index('Microcystis'),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

# x= get_microcontroller_dicts(directory)


from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train", "test"]:
    DatasetCatalog.register("microcontroller_" + d,
                            lambda d=d: get_microcontroller_dicts(directory))
    MetadataCatalog.get("microcontroller_" + d).set(
        thing_classes=['Microcystis'])
microcontroller_metadata = MetadataCatalog.get("microcontroller_train")

from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.model_zoo import model_zoo


# cfg_file = pkg_resources.resource_filename(
#         "detectron2.model_zoo", os.path.join("configs", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")


cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/R_50_1x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("microcontroller_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
# cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-InstanceSegmentation/R_50_1x.yaml")
cfg.SOLVER.IMS_PER_BATCH = 3
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = '/mnt/ssm/AdelaiDet/detectron2/1005_annotation/4'

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()


###############

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.3
cfg.DATASETS.TEST = ("microcontroller_train", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode, Visualizer
dataset_dicts = get_microcontroller_dicts(directory)

import random
import cv2
import matplotlib.pylab as plt

for d in random.sample(dataset_dicts, 5):
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
                   metadata=microcontroller_metadata,
                   scale=0.8,
                   instance_mode=ColorMode.IMAGE_BW   # remove the colors of unsegmented pixels
    )
    v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    plt.figure(figsize = (14, 10))
    plt.imshow(cv2.cvtColor(v.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB))
    plt.show()