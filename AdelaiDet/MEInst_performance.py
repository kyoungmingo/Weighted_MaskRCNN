import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import numpy as np
import os

#test file loading
def test_microcontroller_dicts(directory):
    dataset_dicts = []
    for filename in [file for file in os.listdir(directory) if file.endswith('.png')]:
        filename = os.path.join(directory, filename)
        record = {}

        record["file_name"] = filename
        record["height"] = 512
        record["width"] = 512

        dataset_dicts.append(record)
    return dataset_dicts


from detectron2.engine import DefaultTrainer, DefaultPredictor
from adet.config import get_cfg
from detectron2.model_zoo import model_zoo

cfg = get_cfg()
# cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/MEInst_R_50_1x.yaml"))
cfg.DATASETS.TRAIN = ("microcontroller_train",)
cfg.DATASETS.TEST = ("microcontroller_test",)
cfg.DATALOADER.NUM_WORKERS = 2
cfg.SOLVER.IMS_PER_BATCH = 4
cfg.SOLVER.BASE_LR = 0.00025
cfg.SOLVER.MAX_ITER = 5000
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
cfg.OUTPUT_DIR = '/mnt/ssm/AdelaiDet/1123/6'
cfg.MODEL.WEIGHTS = os.path.join(cfg.OUTPUT_DIR, "model_final.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
predictor = DefaultPredictor(cfg)
dataset_dicts = test_microcontroller_dicts('/mnt/ssm/circle/test/6')

from detectron2.utils.visualizer import ColorMode, Visualizer
import random
import cv2
import matplotlib.pylab as plt

for d in random.sample(dataset_dicts, 1):
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





for d in dataset_dicts:
    im = cv2.imread(d["file_name"])
    outputs = predictor(im)

    pred_out = d["file_name"].split('/')[-1].split('.')[0]
    y = np.asarray(outputs["instances"].pred_masks.to("cpu"))

    # array 저장 npz확장자로
    np.savez_compressed('/mnt/ssm/MEI_circle/6/'+ pred_out, y=y)

# allow_pickle False Error가 남 bbox에서
# load = np.load('/mnt/ssm/elipse/samplexyz.npz', allow_pickle=True)

###inference time##
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
time = []

for i in np.arange(10):
    start.record()
    for d in np.arange(10):
        im = cv2.imread(dataset_dicts[d]["file_name"])
        outputs = predictor(im)

    end.record()

    # Waits for everything to finish running
    torch.cuda.synchronize()

    print(start.elapsed_time(end))

    time.append(start.elapsed_time(end))

np.mean(time)
