import os
import numpy as np
import torch
from PIL import Image
import transforms as T
from engine import train_one_epoch, evaluate
import natsort
import utils

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3"

#file 확장자 지정 Data 불러오기
EXTENSIONS_LABEL = ['.npy']
EXTENSIONS_IMAGE = ['.png']
# EXTENSIONS_IMAGE = ['.jpg']

def is_label(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_LABEL)

def is_image(filename):
    return any(filename.endswith(ext) for ext in EXTENSIONS_IMAGE)

class TrainDataset(object):
    def __init__(self,directory, transforms):
        #         self.root = root
        self.transforms = transforms
        # 모든 이미지 파일들을 읽고, 정렬하여
        # 이미지와 분할 마스크 정렬을 확인합니다

        # placeholder for filenames
        self.imgs = []
        self.masks = []

        # get paths for each

        # label_path = os.path.join("/mnt/ssm/PennFudanPed/4/train")
        # label_path = os.path.join("/mnt/ssm/circle/train/6")
        label_path = os.path.join("/mnt/ssm/algae_dataset/algae_total/train/npy")
        # label_path = os.path.join("/mnt/ssm/membrane/train/npy")

        # image_path = os.path.join("/mnt/ssm/PennFudanPed/4/train")
        image_path = os.path.join(directory)

        # get files
        label_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(label_path)) for f in fn if is_label(f)]
        #이렇게 하면, 하위 폴더 내의 것들도 다 불러와진다.
        image_files = [os.path.join(dp, f) for dp, dn, fn in os.walk(
            os.path.expanduser(image_path)) for f in fn if is_image(f)]

        # sort 1,10,11,...
        # natsort 1,2,3,...
        label_files = natsort.natsorted(label_files)
        image_files = natsort.natsorted(image_files)

        self.imgs.extend(image_files)
        self.masks.extend(label_files)

    def __getitem__(self, idx):
        # 이미지와 마스크를 읽어옵니다
        img_path = self.imgs[idx]
        mask_path = self.masks[idx]
        img = Image.open(img_path).convert("RGB")
        # 분할 마스크는 RGB로 변환하지 않음을 유의하세요
        # 왜냐하면 각 색상은 다른 인스턴스에 해당하며, 0은 배경에 해당합니다
        mask = np.load(mask_path, allow_pickle=True)
        # numpy 배열을 PIL 이미지로 변환합니다
        #         mask = np.array(mask)
        # 인스턴스들은 다른 색들로 인코딩 되어 있습니다.
        obj_ids = np.unique(mask)
        #         # 첫번째 id 는 배경이라 제거합니다
        obj_ids = obj_ids[1:]
        # 컬러 인코딩된 마스크를 바이너리 마스크 세트로 나눕니다
        masks = mask == obj_ids[:, None, None]

        # 각 마스크의 바운딩 박스 좌표를 얻습니다
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            # xmin2 = np.min(xmin,xmax)
            # xmax2 = np.max(xmin, xmax)
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # ymin2 = np.min(ymin,ymax)
            # ymax2 = np.max(ymin,ymax)
            boxes.append([xmin, ymin, xmax, ymax])

        # 모든 것을 torch.Tensor 타입으로 변환합니다
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # 객체 종류는 한 종류만 존재합니다(역자주: 예제에서는 사람만이 대상입니다)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        name = self.imgs[idx].split('/')[-1].split('.')[0]
        image_id = torch.tensor([idx])
        image_name = name

        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # 모든 인스턴스는 군중(crowd) 상태가 아님을 가정합니다
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #test 결과 뽑을 때, image naming을 위함
        # target["image_name"] = image_name

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.imgs)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor


def get_model_instance_segmentation(num_classes):
    # COCO 에서 미리 학습된 인스턴스 분할 모델을 읽어옵니다
    model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)

    # 분류를 위한 입력 특징 차원을 얻습니다
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # 미리 학습된 헤더를 새로운 것으로 바꿉니다
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # 마스크 분류기를 위한 입력 특징들의 차원을 얻습니다
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    hidden_layer = 256
    # 마스크 예측기를 새로운 것으로 바꿉니다
    model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                       hidden_layer,
                                                       num_classes)

    return model


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# 도움 함수를 이용해 모델을 가져옵니다
model = get_model_instance_segmentation(2)

# 모델을 GPU나 CPU로 옮깁니다
model.to(device)

dataset = TrainDataset('/mnt/ssm/algae_dataset/algae_total/train/resized_image',get_transform(train=False))
dataset_test = TrainDataset('/mnt/ssm/algae_dataset/algae_total/train/resized_image',get_transform(train=False))

indices = torch.randperm(len(dataset)).tolist()

dataset = torch.utils.data.Subset(dataset, indices[:-150])
dataset_test = torch.utils.data.Subset(dataset_test, indices[-150:])

data_loader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=4,
    collate_fn=utils.collate_fn)
data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=4,
        collate_fn=utils.collate_fn)

# 옵티마이저(Optimizer)를 만듭니다
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005,
                            momentum=0.9, weight_decay=0.0005)
# 학습률 스케쥴러를 만듭니다
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                               step_size=3,
                                               gamma=0.1)

#weighted maskr
#기존 maskr transfer learning을 위해 maskr weight load
model.load_state_dict(torch.load('/mnt/ssm/weighted_microcyistis/mask_model.pth'))

num_epochs = 5

for epoch in range(num_epochs):
    # 1 에포크동안 학습하고, 10회 마다 출력합니다
    train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
    # 학습률을 업데이트 합니다
    lr_scheduler.step()
    # 테스트 데이터셋에서 평가를 합니다
    # evaluate(model, data_loader, device=device)
    evaluate(model, data_loader_test, device=device)

print("That's it!")

torch.save(model.state_dict(), '/mnt/ssm/weighted_microcyistis/Wmask_model3.pth')


# pick one image from the test set
img, label = dataset_test[0]

# put the model in evaluation mode
model.eval()
with torch.no_grad():
    prediction = model([img.to(device)])

import matplotlib.pyplot as plt

plt.imshow(img.mul(255).permute(1, 2, 0))
plt.show()
pred=torch.sum(prediction[0]['masks'],dim=0).squeeze(0)
# pred=prediction[0]['masks'][0,0] +prediction[0]['masks'][1,0]+prediction[0]['masks'][2,0]
# +prediction[0]['masks'][3,0]+prediction[0]['masks'][4,0]+prediction[0]['masks'][5,0]\
     # +prediction[0]['masks'][6,0]+prediction[0]['masks'][7,0]
# probability threshold
pred[pred>=0.5]=1
pred[pred<0.5]=0
plt.imshow(pred.mul(255).cpu())
plt.show()

#test 결과 뽑기
model.load_state_dict(torch.load('/mnt/ssm/weightedmask_circle/62/Wmask_model.pth'))
model.eval()

#measure 측정을 위한, prediction mask save
for d in np.arange(len(dataset_test)):
    img, _ = dataset_test[d]
    with torch.no_grad():
        prediction = model([img.to(device)])
        # prediction = model([img])

    y = prediction[0]['masks'].cpu()
    name=str(int(dataset_test[d][1]['image_name']))
    # array 저장 npz확장자로
    np.savez_compressed('/mnt/ssm/weightedmask_circle/62/'+ name, y=y)

#inference time 측정
start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)

start.record()
for d in np.arange(10):
    img, _ = dataset_test[d]
    with torch.no_grad():
        prediction = model([img.to(device)])
end.record()

# Waits for everything to finish running
torch.cuda.synchronize()

print(start.elapsed_time(end))

#https://github.com/warmspringwinds/pytorch-segmentation-detection/blob/master/pytorch_segmentation_detection/metrics.py
def iou(pred, target):
    ious = []
    pred = pred.view(-1)
    target = target.view(-1)
  # Ignore IoU for background class ("0")
  # for cls in xrange(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == 1
    target_inds = target == 1
    intersection = (pred_inds[target_inds]).long().sum()  # Cast to long to prevent overflows
    union = pred_inds.long().sum() + target_inds.long().sum() - intersection
    if union == 0:
      ious.append(float('nan'))  # If there is no ground truth, do not include in evaluation
    else:
      ious.append(float(intersection) / float(max(union, 1)))
    return np.array(ious)

miou = []
for i in range(len(dataset_test)):

    img, lab = dataset_test[i]

    model.eval()

    with torch.no_grad():
        prediction = model([img.to(device)])
        prediction = prediction[0]['masks'].squeeze(dim=1)
        pred = prediction[0:len(lab['labels'])]>=0.5
        pred = torch.sum(pred, dim=0).squeeze(0)
        pred = pred.cpu().int()

    target = lab['masks']
    target = torch.sum(target, dim=0).squeeze(0)
    target = target.cpu().int()
    print(i)
    miou.append(iou(pred,target))