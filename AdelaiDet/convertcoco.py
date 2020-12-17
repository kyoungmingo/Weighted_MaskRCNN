#https://github.com/TannerGilbert/Detectron2-Train-a-Instance-Segmentation-Model
#for MEInst >> detectron2
import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1,2,3"

import os
import numpy as np
import json
from detectron2.structures import BoxMode

def get_microcontroller_dicts(directory):
    classes = ['circle']
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory + 'json/') if file.endswith('.json')]:
        json_file = os.path.join(directory,'json', filename)
        with open(json_file) as f:
            img_anns = json.load(f)

        record = {}

        # filename = os.path.join(directory, img_anns["imagePath"])
        #json file 내의 imagePath를 수정하여 filename 지정!
        filename = os.path.join(directory, img_anns['imagePath'].split('\\')[-1])

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


from detectron2.data import DatasetCatalog, MetadataCatalog

for d in ["train", "test"]:
    DatasetCatalog.register("microcontroller_" + d,
                            # lambda d=d: get_microcontroller_dicts('/mnt/ssm/algae_dataset/algae_total/' + d + '/'))
                            # lambda d=d: get_microcontroller_dicts('/mnt/ssm/elipse/annotation/4/' + d + '/' ))
                            lambda d=d: get_microcontroller_dicts('/mnt/ssm/circle/' + d + '/6/'))

    MetadataCatalog.get("microcontroller_" + d).set(
            thing_classes=['Microcystis'])
microcontroller_metadata = MetadataCatalog.get("microcontroller_train")

from detectron2.engine import DefaultTrainer, DefaultPredictor
#mask rcnn training시 detectron2의 config 활용
# from detectron2.config import get_cfg
#MEInst의 경우 adet의 config 활용
from adet.config import get_cfg
from detectron2.model_zoo import model_zoo

# from adet.checkpoint import AdetCheckpointer
# cfg_file = pkg_resources.resource_filename(
#         "detectron2.model_zoo", os.path.join("configs", "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/MEInst_R_50_1x.yaml"))
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATASETS.TRAIN = ("microcontroller_train",)
cfg.DATASETS.TEST = ()
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = '/mnt/ssm/AdelaiDet/1120/MEInst_R_50_1x.pth'
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 3000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = '/mnt/ssm/AdelaiDet/test'

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
dataset_dicts = get_microcontroller_dicts('/mnt/ssm/circle/train/6/')


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

outputs = predictor(im)