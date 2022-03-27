import torch
from util.yolo_result import *
import os
from model import Mini_model
import json
import cv2
class Train():
    def __init__(self):
        self.model = Mini_model()
        self.device = 'cuda:1'
        return
    def train(self, yoloResult):
        target_path = "/workspace/data/Obb/obb_gt.json"
        target_json = list(json.load(open(target_path)).values())
        # criterion = torch.nn.NLLLoss(reduction="mean").to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000005)

        self.model = self.model.to(self.device)
        self.model.train()
        lossFunction = torch.nn.MSELoss()
        # print("HERE 1")
        min_loss = 9999999
        best_epoch = 0
        log = open("./weights/log.txt", "w")

        for epoch in range(70000):
            # print("HERE 2")

            mean_loss = 0
            count = 0
            non = 0
            # Search all image
            for img in yoloResult.images:
                input_image = cv2.imread(img.fileDir)
                # print(input_image.shape)
                # print(img.fileDir)

                for item in img.item:
                    count +=1
                    flag = False

                    for instance in target_json[img.fileName]["Objects"]:
                        # Find object from gt file
                        if instance["model"] == yoloResult.classes[item.class_num]:
                            # print("FIND!! {}".format(instance["model"]))
                            gt = instance["projection point"]
                            flag = True
                    if not flag:
                        continue
                    # print("FLAG!!!! {}".format(flag))
                    # print("item bbox = {}".format(item))
                    lx = item.bbox[0] if item.bbox[0] > 0 else 0
                    rx = (item.bbox[0] + item.bbox[2]) if (item.bbox[0] + item.bbox[2]) < input_image.shape[1] else input_image.shape[1]
                    ty = item.bbox[1] if item.bbox[1] > 0 else 0
                    by = (item.bbox[1] + item.bbox[3]) if (item.bbox[1] + item.bbox[3]) < input_image.shape[0] else input_image.shape[0]
  
                    gt = np.array(list(gt[0].values()))
                    # print(gt)
                    gt = np.reshape(gt[:4,:2], (8))
                    # print(gt)

                    target = yoloResult.normal_obbpoint(gt, [lx, rx, ty, by]) 
                    target = torch.Tensor(target)
                    # print("     rounded bbox {} {} {} {}".format(lx, ty, rx, by))

                    crop_image = input_image[ty:by,lx:rx,:]
                    crop_image = cv2.resize(crop_image, dsize=(120, 120), interpolation = cv2.INTER_CUBIC)
                    crop_tensor = torch.unsqueeze(torch.from_numpy(crop_image).permute(2,0,1), 0).to(self.device).float()

                    # crop_tensor = crop_tensor[:,:,ty:by,lx:rx]
                    # print("Crop shape {}".format(crop_tensor.shape))
                    output = self.model(crop_tensor)

                    target= torch.unsqueeze(target, 0).to(self.device)

                    # print(target.shape)
                    # print(output.shape)
                    loss = lossFunction(output, target)
                    loss.backward()
                    mean_loss += loss/222
                    optimizer.step()
            print("epoch = {} mean_loss {}, best_epoch = {}, best_loss = {}".format(epoch, mean_loss, best_epoch, min_loss))
            log.write(str(mean_loss.item()))
            log.write("\n")
            if epoch>0 and epoch%100==0:
                file_path = "./weights" +'/r_epoch_'+str(epoch) + '.pth'
                torch.save(self.model.state_dict(),file_path)             
            if epoch>0 and epoch%10==0 and min_loss > mean_loss:
                file_path = "./weights" +'/best'+ '.pth'
                torch.save(self.model.state_dict(),file_path)  
                min_loss = mean_loss
                best_epoch = epoch
            # print("Count {}".format(count))                    

                    # gt_obb_points = 

                    # print("     Target = {}".format(target[img.fileName]))

                # return

                # print("HERE 3")
            #     print("DATASET IDX = {}".format(data_set))
            #     for index, (data, target) in enumerate(data_loader):
            #         optimizer.zero_grad()  # gradient init
            #         output = F.log_softmax(self.model(data[:,:,:,400:800]), dim=1)
            #         loss = criterion(output, target.long()[:,:,400:800])
            #         loss.backward()  # backProp
            #         optimizer.step()
            #         self.loss = loss.item()
            # if epoch % 5 == 0:
            #     print("LOG!!")
                # self.logger.logging(self)


    # normal   = world frame -> object frame
    # unnormal = object frame -> world frame
