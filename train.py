from ObbDataset import ObbDataset
import torch
import torchvision
import torchvision.models
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
import glob
from engine import train_one_epoch, evaluate
import utils
import os
class Train():
    def __init__(self):
        return
    def train(self):
        print("Training---------")
        device= torch.device('cpu')

        num_classes = 5

        dataset = ObbDataset()
        data_loader = torch.utils.data.DataLoader( dataset, batch_size=1, shuffle=True, num_workers=1)

        model = self.get_model_instance_segmentation(num_classes)
        model.to(device)
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

        print(len(data_loader))
        for img, target in data_loader:
            # print(img)
            # print(target)
            continue
        # print(data_loader[0])
        return
        num_epochs=10
        for epoch in range(num_epochs):
            # 1 에포크동안 학습하고, 10회 마다 출력합니다
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
            # 학습률을 업데이트 합니다
            lr_scheduler.step()
            # 테스트 데이터셋에서 평가를 합니다
            # evaluate(model, data_loader_test, device=device)
        # data_loader = torch.utils.data.DataLoader( dataset, batch_size=2, shuffle=True, num_workers=1, collate_fn=utils.collate_fn)
        # file_list = dataset.get_traindata_list(dir_path)


    def get_model_instance_segmentation(self, num_classes):
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
        model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask, hidden_layer, num_classes)
        return model