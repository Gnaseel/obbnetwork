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
        target_path = "/workspace/data/Obb/2022-03-28_OBB_640480_2000frame/obb_gt.json"
        target_json = list(json.load(open(target_path)).values())
        # criterion = torch.nn.NLLLoss(reduction="mean").to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.000005)

        self.model = self.model.to(self.device)
        self.model.train()
        # lossFunction = torch.nn.MSELoss()
        lossFunction = torch.nn.L1Loss()
        # print("HERE 1")
        min_loss = 9999999
        best_epoch = 0
        # log = open("./weights/speed/log2.txt", "w")
        log = open("./weights/data2000/log2.txt", "w")
        total_item_count = yoloResult.get_item_count()
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
  
                    gt = np.array(list(gt[0].values()))[:,:2]
                    gt = self.get_target4Points(gt)
                    gt = np.reshape(gt, (8))

                    target = yoloResult.normal_obbpoint(gt, [lx, rx, ty, by]) 
                    target = torch.Tensor(target)

                    crop_image = input_image[ty:by,lx:rx,:]
                    crop_image = cv2.resize(crop_image, dsize=(120, 120), interpolation = cv2.INTER_CUBIC)
                    crop_tensor = torch.unsqueeze(torch.from_numpy(crop_image).permute(2,0,1), 0).to(self.device).float()

                    output = self.model(crop_tensor)

                    target= torch.unsqueeze(target, 0).to(self.device)


                    loss = lossFunction(output[:6], target)
                    # print("Target = {}".format(target))
                    # print("Output = {}".format(output))
                    # print("LOSS = {}".format(loss))
                    loss.backward()
                    mean_loss += loss/total_item_count
                    optimizer.step()
            print("epoch = {} mean_loss {}, best_epoch = {}, best_loss = {}".format(epoch, mean_loss, best_epoch, min_loss))
            # print(str(mean_loss.item()))
            log.write(str(mean_loss.item()))
            log.write("\n")
            if epoch>0 and epoch%100==0:
                file_path = "./weights/speed" +'/r_epoch_'+str(epoch) + '.pth'
                torch.save(self.model.state_dict(),file_path)             
            if epoch>0 and epoch%5==0 and min_loss > mean_loss:
                file_path = "./weights/speed" +'/best'+ '.pth'
                torch.save(self.model.state_dict(),file_path)  
                min_loss = mean_loss
                best_epoch = epoch

    def get_target4Points(self, gt):
        re_array = np.zeros((4,2))
        # re_array[0] = low = gt[np.argmin(gt[:,1])]
        re_array[0] = high = gt[np.argmax(gt[:,1])]
        re_array[1] = left = gt[np.argmin(gt[:,0])]
        re_array[2] = right = gt[np.argmax(gt[:,0])]
        # new_list = low+
        # print(re_array)
        return re_array

    # normal   = world frame -> object frame
    # unnormal = object frame -> world frame
