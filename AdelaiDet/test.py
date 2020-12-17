
import os
import numpy as np
import json
from detectron2.structures import BoxMode

directory = "/mnt/ssm/elipse/annotation/4"
def get_microcontroller_dicts(directory):
    classes = ['Microcystis']
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.json')]:
        json_file = os.path.join(directory1, filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        filename = os.path.join(directory1, img_anns["imagePath"])

        record["file_name"] = filename
        record["height"] = 512
        record["width"] = 512

        annos = img_anns["shapes"]
        objs = []
        for anno in annos:
            px = [a[0] for a in anno['points']]
            py = [a[1] for a in anno['points']]
            poly = [(x, y) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": classes.index(anno['label']),
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


import matplotlib.pyplot as plt
plt.scatter(px, py)
plt.show()


x =np.asarray(outputs["instances"][0].pred_boxes)
y = np.asarray(outputs["instances"][0].pred_masks.to("cpu"))
z = np.asarray(outputs["instances"][0].scores.to("cpu"))


#array 저장 npz확장자로
np.savez_compressed('/mnt/ssm/elipse/samplexyz',x=x,y=y,z=z)

#allow_pickle False Error가 남 bbox에서
load = np.load('/mnt/ssm/elipse/samplexyz.npz', allow_pickle=True)

#test file loading
def test_microcontroller_dicts(directory):

    dataset_dicts = []


    for filename in [file for file in os.listdir(directory) if file.endswith('.png')]:
        filename = os.path.join(directory, filename)
        record = {}

        record["file_name"] = filename
        record["height"] = 512
        record["width"] = 512

        # annos = img_anns["shapes"]
        # objs = []
        # for anno in annos:
        #     px = [a[0] for a in anno['points']]
        #     py = [a[1] for a in anno['points']]
        #     poly = [(x, y) for x, y in zip(px, py)]
        #     poly = [p for x in poly for p in x]
        #
        #     obj = {
        #         "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
        #         "bbox_mode": BoxMode.XYXY_ABS,
        #         "segmentation": [poly],
        #         "category_id": classes.index(anno['label']),
        #         "iscrowd": 0
        #     }
        #     objs.append(obj)
        # record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts


for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    x = np.asarray(outputs["instances"][0].pred_boxes)
    y = np.asarray(outputs["instances"][0].pred_masks.to("cpu"))
    z = np.asarray(outputs["instances"][0].scores.to("cpu"))

    # array 저장 npz확장자로
    np.savez_compressed('/mnt/ssm/elipse/samplexyz', x=x, y=y, z=z)

    # allow_pickle False Error가 남 bbox에서
    load = np.load('/mnt/ssm/elipse/samplexyz.npz', allow_pickle=True)









from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train", "test"]:
    DatasetCatalog.register("microcontroller_" + d,
                            lambda d=d: get_microcontroller_dicts(directory1))
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
cfg.SOLVER.IMS_PER_BATCH = 1
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 100
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = '/mnt/ssm/AdelaiDet/detectron2/0925_example'

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = DefaultTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

#####training _ Detectron2#####

cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.DATASETS.TEST = ("microcontroller_train", )
predictor = DefaultPredictor(cfg)

from detectron2.utils.visualizer import ColorMode, Visualizer
dataset_dicts = get_microcontroller_dicts(directory1)

import random
import cv2
import matplotlib.pylab as plt

for d in random.sample(dataset_dicts, 3):
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