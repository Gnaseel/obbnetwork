from ObbDataset import ObbDataset
import torch
import transforms as T
import torchvision
import torchvision.models
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import glob
from engine import train_one_epoch, evaluate
import utils
import os
class Train():
    def __init__(self):
        return
    def train(self):
        device = 'cpu'
        dataset = ObbDataset()
        # 데이터셋을 학습용과 테스트용으로 나눕니다(역자주: 여기서는 전체의 50개를 테스트에, 나머지를 학습에 사용합니다)
        indices = torch.randperm(len(dataset)).tolist()
        dataset = torch.utils.data.Subset(dataset, indices[:-50])

        # 데이터 로더를 학습용과 검증용으로 정의합니다
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=2, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)


        num_classes = 5
        # 도움 함수를 이용해 모델을 가져옵니다
        model = self.get_model_instance_segmentation(num_classes)

        # 모델을 GPU나 CPU로 옮깁니다
        model.to(device)

        # 옵티마이저(Optimizer)를 만듭니다
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                    momentum=0.9, weight_decay=0.0005)
        # 학습률 스케쥴러를 만듭니다
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                       step_size=3,
                                                       gamma=0.1)

        # 10 에포크만큼 학습해봅시다
        num_epochs = 10

        for epoch in range(num_epochs):
            # 1 에포크동안 학습하고, 10회 마다 출력합니다
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # 학습률을 업데이트 합니다
            lr_scheduler.step()
            # 테스트 데이터셋에서 평가를 합니다
            

    def get_model_instance_segmentation(self, num_classes):
            # load an instance segmentation model pre-trained on COCO
        model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    
        # get number of input features for the classifier
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        # replace the pre-trained head with a new one
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
        # now get the number of input features for the mask classifier
        in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
        hidden_layer = 256
        # and replace the mask predictor with a new one
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                           hidden_layer,
                                                           num_classes)
    
        return model
    def train_test(self):
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        dataset = ObbDataset('D:\Devel\Obb\Dataset', self.get_transform(train=True))
        data_loader = torch.utils.data.DataLoader(
         dataset, batch_size=2, shuffle=True, num_workers=4,
         collate_fn=utils.collate_fn)
        # 학습 시
        images,targets = next(iter(data_loader))
        images = list(image for image in images)
        targets = [{k: v for k, v in t.items()} for t in targets]
        output = model(images,targets)   # Returns losses and detections
        # 추론 시
        model.eval()
        x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        predictions = model(x)           # Returns predictions
        return

    def get_transform(self, train):
        transforms = []
        transforms.append(T.ToTensor())
        if train:
            # (역자주: 학습시 50% 확률로 학습 영상을 좌우 반전 변환합니다)
            transforms.append(T.RandomHorizontalFlip(0.5))
        return T.Compose(transforms)