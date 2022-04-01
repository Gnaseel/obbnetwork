from ObbDataset import ObbDataset
import torch
from model import Mini_model
import numpy as np
import sampleColor as myColor
import json
import cv2
import torchvision
import os
import time
from timeit import default_timer as timer
class Inference():
    def __init__(self):
        self.device = 'cuda:0'
        return
    def inference(self, mode_path, yoloResult):
        target_path = "/workspace/data/Obb/2022-03-28_OBB_640480_2000frame/obb_gt.json"
        target_json = list(json.load(open(target_path)).values())
        num_classes = 4
        self.model = Mini_model().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(mode_path, "best.pth"), map_location='cpu'))
        self.model.eval()
        total_inference_time = 0
        for img in yoloResult.images:
            input_image = cv2.imread(img.fileDir)
            gt_image = cv2.imread(img.fileDir)
            img_inference_time = 0
            for idx, item in enumerate(img.item):
                # print(item)
                for instance in target_json[img.fileName]["Objects"]:
                    # Find object from gt file
                    if instance["model"] == yoloResult.classes[item.class_num]:
                        # print("FIND!! {}".format(instance["model"]))
                        gt = instance["projection point"]
                        gt = np.array(list(gt[0].values()))[:,:2]
                        gt = np.reshape(gt, (8,2))
                        gt = gt.astype(np.int64)

                lx = item.bbox[0] if item.bbox[0] > 0 else 0
                rx = (item.bbox[0] + item.bbox[2]) if (item.bbox[0] + item.bbox[2]) < input_image.shape[1] else input_image.shape[1]
                ty = item.bbox[1] if item.bbox[1] > 0 else 0
                by = (item.bbox[1] + item.bbox[3]) if (item.bbox[1] + item.bbox[3]) < input_image.shape[0] else input_image.shape[0]

                # target = self.getGTobbPoints(gt, [lx, rx, ty, by]) 
                crop_image = input_image[ty:by,lx:rx,:]
                crop_image = cv2.resize(crop_image, dsize=(120, 120), interpolation = cv2.INTER_CUBIC)
                crop_tensor = torch.unsqueeze(torch.from_numpy(crop_image).permute(2,0,1), 0).to(self.device).float()
                
                inference_start = timer()
                output = self.model(crop_tensor)
                inference_end = timer()
                img_inference_time += inference_end-inference_start
                # print("Input = {}".format(item))
                output =output.detach().cpu().numpy()[0]
                # print("     Output = {} {} {} {} {} {} {} {}".format(*list(output)))

                target = yoloResult.unnormal_obbpoint(list(output), [lx, rx, ty, by]) 

                # print("     POST Output = {} {} {} {} {} {} {} {}".format(*list(target)))
                ceil_dist = by-ty
                point = self.pointParsing(target, ceil_dist)

                input_image = self.draw_img_v2(input_image, point, item.class_num)
                gt_image = self.draw_img_v2(gt_image, gt, item.class_num)
            # print("Infer time {}".format(img_inference_time))
            total_inference_time+=img_inference_time
            total_yolo_time = img.infertime

            cv2.imwrite("./data/"+str(img.fileName)+".jpg", input_image)
            cv2.imwrite("./data/abc.jpg", input_image)
            cv2.imwrite("./data/gt.jpg", gt_image)
            time.sleep(1.0)
        mean_inference_time = total_inference_time/len(yoloResult.images)
        mean_yolo_time = total_yolo_time/len(yoloResult.images)
        print("Total time = {}".format(mean_inference_time+mean_yolo_time))
        print("Mean infer time = {}".format(mean_inference_time))
        print("Mean YOLO time = {}".format(mean_yolo_time))
        return

    def pointParsing(self, target, ceil_dist):
        point = np.zeros((8,2), dtype=np.int64)
        point[0] = [int(target[2]), int(target[3])]
        point[1] = [int(target[4]), int(target[5])]
        point[2] = [int(target[6]), int(target[7])]
        point[3] = [point[1,0]+point[2,0]-point[0,0], point[1,1]+point[2,1]-point[0,1]]
        point[4] = [point[0,0] ,point[0,1]- ceil_dist*0.7]
        point[5] = [point[1,0] ,point[1,1]- ceil_dist*0.7]
        point[6] = [point[2,0] ,point[2,1]- ceil_dist*0.7]
        point[7] = [point[3,0] ,point[3,1]- ceil_dist*0.7]
        return point

    def draw_img_v2(self, input_image, point, color_idx=0):
        font =  cv2.FONT_HERSHEY_PLAIN
        font_size=3
        for idx in range(8):
            input_image = cv2.circle(input_image, (point[idx,0], point[idx,1]), 5, myColor.color_list[color_idx], -1)
            input_image = cv2.putText(input_image, str(idx), (point[idx,0], point[idx,1]), font, font_size, myColor.color_list[color_idx], 5)

        # # Print Line
        input_image = cv2.line(input_image,  (tuple(point[0])), (tuple(point[1])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[0])), (tuple(point[2])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[3])), (tuple(point[1])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[3])), (tuple(point[2])), myColor.color_list[color_idx], 3)

        input_image = cv2.line(input_image,  (tuple(point[0])), (tuple(point[4])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[1])), (tuple(point[5])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[2])), (tuple(point[6])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[3])), (tuple(point[7])), myColor.color_list[color_idx], 3)

        input_image = cv2.line(input_image,  (tuple(point[4])), (tuple(point[5])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[4])), (tuple(point[6])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[7])), (tuple(point[5])), myColor.color_list[color_idx], 3)
        input_image = cv2.line(input_image,  (tuple(point[7])), (tuple(point[6])), myColor.color_list[color_idx], 3)
        return input_image

    def draw_img(self, input_image, item, target, ceil_dist):
        # Print Points
        input_image = cv2.circle(input_image, (int(target[0]),int(target[1])), 5, myColor.color_list[item.class_num], -1)
        input_image = cv2.circle(input_image, (int(target[2]),int(target[3])), 5, myColor.color_list[item.class_num], -1)
        input_image = cv2.circle(input_image, (int(target[4]),int(target[5])), 5, myColor.color_list[item.class_num], -1)
        input_image = cv2.circle(input_image, (int(target[6]),int(target[7])), 5, myColor.color_list[item.class_num], -1)

        input_image = cv2.circle(input_image, (int(target[0]),int(target[1] - (ceil_dist)*0.7)), 5, myColor.color_list[item.class_num], -1)
        input_image = cv2.circle(input_image, (int(target[2]),int(target[3] - (ceil_dist)*0.7)), 5, myColor.color_list[item.class_num], -1)
        input_image = cv2.circle(input_image, (int(target[4]),int(target[5] - (ceil_dist)*0.7)), 5, myColor.color_list[item.class_num], -1)
        input_image = cv2.circle(input_image, (int(target[6]),int(target[7] - (ceil_dist)*0.7)), 5, myColor.color_list[item.class_num], -1)
        # Print Text
        font =  cv2.FONT_HERSHEY_PLAIN
        font_size=3
        input_image = cv2.putText(input_image, "0", (int(target[0]),int(target[1])), font, font_size, myColor.color_list[item.class_num], 5)
        input_image = cv2.putText(input_image, "1", (int(target[2]),int(target[3])), font, font_size, myColor.color_list[item.class_num], 5)
        input_image = cv2.putText(input_image, "2", (int(target[4]),int(target[5])), font, font_size, myColor.color_list[item.class_num], 5)
        input_image = cv2.putText(input_image, "3", (int(target[6]),int(target[7])), font, font_size, myColor.color_list[item.class_num], 5)

        input_image = cv2.putText(input_image, "4", (int(target[0]),int(target[1] - (ceil_dist)*0.7)), font, font_size, myColor.color_list[item.class_num], 5)
        input_image = cv2.putText(input_image, "5", (int(target[2]),int(target[3] - (ceil_dist)*0.7)), font, font_size, myColor.color_list[item.class_num], 5)
        input_image = cv2.putText(input_image, "6", (int(target[4]),int(target[5] - (ceil_dist)*0.7)), font, font_size, myColor.color_list[item.class_num], 5)
        input_image = cv2.putText(input_image, "7", (int(target[6]),int(target[7] - (ceil_dist)*0.7)), font, font_size, myColor.color_list[item.class_num], 5)

        # Print Line
        input_image = cv2.line(input_image,  ( int(target[0]),int(target[1])), (int(target[2]),int(target[3])), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[0]),int(target[1])), (int(target[4]),int(target[5])), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[2]),int(target[3])), (int(target[6]),int(target[7])), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[4]),int(target[5])), (int(target[6]),int(target[7])), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[0]),int(target[1] - (ceil_dist)*0.7)), (int(target[2]),int(target[3] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[0]),int(target[1] - (ceil_dist)*0.7)), (int(target[4]),int(target[5] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[6]),int(target[7] - (ceil_dist)*0.7)), (int(target[2]),int(target[3] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[6]),int(target[7] - (ceil_dist)*0.7)), (int(target[4]),int(target[5] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[0]),int(target[1])), (int(target[0]),int(target[1] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[2]),int(target[3])), (int(target[2]),int(target[3] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[4]),int(target[5])), (int(target[4]),int(target[5] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        input_image = cv2.line(input_image,  ( int(target[6]),int(target[7])), (int(target[6]),int(target[7] - (ceil_dist)*0.7)), myColor.color_list[item.class_num], 3)
        return input_image
