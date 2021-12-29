------------
# Weighted Mask R-CNN for Improving Adjacent Boundary Segmentation

By SungMin Suh, Yongeun Park, KyoungMin Ko, SeongMin Yang, Jaehyeong Ahn, Jae-Ki Shin and SungHwan Kim

This repository contains a Pytorch implementation of Weighted Mask R-CNN with applications to simulated data and real data(e.g., Microcystis, one of the most common algae genera and cell membrane images). The architecture of the proposed model of our Weighted Mask R-CNN can be found below:

<img width="682" alt="스크린샷 2021-12-29 오후 3 24 20" src="https://user-images.githubusercontent.com/35245580/147633418-6bb18aed-c525-4012-aed2-83e533251bcd.png">

Visualization results of real data(Microcystis):

<img width="675" alt="스크린샷 2021-12-29 오후 3 39 07" src="https://user-images.githubusercontent.com/35245580/147634342-af289caa-041f-464b-9e32-dc628dac218d.png">

For more details, please refer to our paper: [Weighted Mask R-CNN](https://www.hindawi.com/journals/js/2021/8872947/). 

# How to Run

Weighted Mask R-CNN is explained with reference to [TORCHVISION OBJECT DETECTION FINETUNING TUTORIAL](https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html).

## Setting

1. To use torchvision tutorial for Mask R-CNN, copy everything under references/detection to your folder and use them here.
2. To use Weighted Mask R-CNN, modify roi_heads.py under your environment's torchvision to vision/torchvision/models/detection/roi_heads.py.
+torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2
+If it is different from our version, copy and paste definitions "getweightmap" & "my_weightmap" on your roi_heads.py.
+Modify the definition "maskrcnn_loss" the same as our "maskrcnn_loss".
3. All data sets are available at the author’s website (http://www.hifiai.pe.kr).

## Training

The easiest way is to open the colab notebook.

## Inference





# Weighted_MaskRCNN

#training
Weighted_maskrcnn.py는 color map label을 통해서 training이 가능하다.

>vision/torchvision/models/detection/roi_heads.py에서 weight map의 range를 delta 값을 수정해줌으로써 변경가능하다.
>해당 파일에 따라서 torchvision의 roi_heads.py를 똑같이 수정해주면, weighted mask rcnn 작동이 가능하다.

#detectron_MEInst

모델 비교를 위해 detectron2 플랫폼을 활용해 다양한 instance segmentation 활용이 가능하다.

>detectron2는 config파일을 통해 다양한 모델 적용이 가능하다.
>MEInst의 config파일을 설치하여 detectron2를 활용하였다.
>detectron2의 경우 coco 기반의 annotation 형식을 지원한다.

>그러므로, 다른 json file 형식이 지원 가능하도록 각각의 box 및 mask 등을 따로 지정하여, 모델 training이 가능하도록 하여 모델 비교를 진행하였다.
>AdelaiDet/convertcoco.py는 위의 방법이 적용된 detectron2 활용 코드이다.

#evaluate
measure 측정은 map, miou, 논문에 나온 instance간의 거리를 prediction과 gt의 차이를 통해 제시한 measure 총 3가지를 비교하였다.

>Weighted_maskrcnn.py의 evaluate을 통해 measure 측정이 가능하다.
>AdelaiDet/MEInst_performance.py 를 통해 distance에 따른 measure 측정을 위한 작업이 진행가능하다.
