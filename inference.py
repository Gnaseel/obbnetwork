from ObbDataset import ObbDataset
import torch
from model import Mini_model
import json
import cv2
import torchvision
import os
class Inference():
    def __init__(self):
        self.device = 'cuda:0'
        return
    def inference(self, mode_path, yoloResult):
        target_path = "/workspace/data/Obb/obb_gt.json"
        target_json = list(json.load(open(target_path)).values())
        num_classes = 4
        self.model = Mini_model().to(self.device)
        self.model.load_state_dict(torch.load(os.path.join(mode_path, "best.pth"), map_location='cpu'))
        self.model.eval()
        for img in yoloResult.images:
            input_image = cv2.imread(img.fileDir)
            # print(input_image.shape)
            # print(img.fileDir)
            for item in img.item:
                # print(item)
                for instance in target_json[img.fileName]["Objects"]:
                    # Find object from gt file
                    if instance["model"] == yoloResult.classes[item.class_num]:
                        # print("FIND!! {}".format(instance["model"]))
                        gt = instance["projection point"]
                lx = item.bbox[0] if item.bbox[0] > 0 else 0
                rx = (item.bbox[0] + item.bbox[2]) if (item.bbox[0] + item.bbox[2]) < input_image.shape[1] else input_image.shape[1]
                ty = item.bbox[1] if item.bbox[1] > 0 else 0
                by = (item.bbox[1] + item.bbox[3]) if (item.bbox[1] + item.bbox[3]) < input_image.shape[0] else input_image.shape[0]

                # target = self.getGTobbPoints(gt, [lx, rx, ty, by]) 
                crop_image = input_image[ty:by,lx:rx,:]
                crop_image = cv2.resize(crop_image, dsize=(120, 120), interpolation = cv2.INTER_CUBIC)
                crop_tensor = torch.unsqueeze(torch.from_numpy(crop_image).permute(2,0,1), 0).to(self.device).float()
                
                output = self.model(crop_tensor)
                print("Input = {}".format(item))
                output =output.detach().cpu().numpy()[0]
                print("     Output = {} {} {} {} {} {} {} {}".format(*list(output)))

                target = yoloResult.unnormal_obbpoint(list(output), [lx, rx, ty, by]) 

                print("     POST Output = {} {} {} {} {} {} {} {}".format(*list(target)))
                # input_image = cv2.line(seg_img, coords[j], coords[j+1], (i+1, i+1, i+1), SEG_WIDTH//2)
                input_image = cv2.circle(input_image, (int(target[0]),int(target[1])), 5, (0,0,255), -1)
                input_image = cv2.circle(input_image, (int(target[2]),int(target[3])), 5, (0,0,255), -1)
                input_image = cv2.circle(input_image, (int(target[4]),int(target[5])), 5, (0,0,255), -1)
                input_image = cv2.circle(input_image, (int(target[6]),int(target[7])), 5, (0,0,255), -1)

                # return

            cv2.imwrite("./data/"+str(img.fileName)+".jpg", input_image)
      
        return