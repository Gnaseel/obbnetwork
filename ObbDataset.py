import glob
import os
import torch
import numpy as np
from PIL import Image
import sampleColor
import json
import cv2
class ObbDataset(torch.utils.data.Dataset):
    def __init__(self, path="D:\Devel\Obb\Dataset", transforms=None):
        # dir_path = glob.glob(os.path.join(path,"19201080_100_multi_OBB"))[0]
        # dir_path = glob.glob(os.path.join(path,"640480_100_multi_OBB"))[0]
        dir_path = glob.glob(os.path.join(path,"22_03_16_obb_640480"))[0]

        self.dir_path = dir_path
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(dir_path, "RGB"))))
        self.masks = list(sorted(os.listdir(os.path.join(dir_path, "mask_all"))))

        json_dir = os.path.join(dir_path, "obb_gt.json")
        self.json_object = json.load(open(json_dir))
        # print(self.json_object)
        print("----------------------------------------------")
        print("dir_path = {}".format(self.dir_path))
        print("Img_path = {}".format(self.imgs))
        print("Mask_path = {}".format(self.masks))
        print("----------------------------------------------") 
        return

    def __getitem__(self, idx):
        # print("----------------------------------------------")
        # load images ad masks
        img_path = os.path.join(self.dir_path, "RGB", self.imgs[idx])
        img_num = img_path.split(os.sep)[-1][:-4]
        data = self.json_object[str(img_num)]["Objects"]
        img_re = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)

        # print("IDX {}".format(idx))
        # print("Img_num = {}".format(img_num))
        # print("Img_path = {}".format(img_path))
        # print("----------------------------------------------")

        # obbPoints=[]
        labels = []
        boxes = []
        target_mask = np.full((480, 640, 3), False)
        # print("")
        # print("IMG = {}".format(img_path))
        for i, instance in enumerate(data):
            instance_labels = instance["model"]
            # print("instance {}".format(instance))
            # print("OBJECT = {}".format(instance_labels))
            modelIdx = self.getModelIdx(instance_labels)
            mask_path = os.path.join(self.dir_path, "mask", str(modelIdx), self.masks[idx])
    
            mask = cv2.imread(mask_path)
            # mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)

            ojb_ids = np.unique(mask)[2:]
            mask2 = mask == ojb_ids[:,None,None]
            target_mask = target_mask + mask2

            pos = np.where(mask2)
            xmin = int(np.min(pos[1]))
            xmax = int(np.max(pos[1]))
            ymin = int(np.min(pos[0]))
            ymax = int(np.max(pos[0]))

            # print("{} {} {} {}".format(xmin,xmax, ymin, ymax))
            # print("")

            boxes.append([xmin, ymin, xmax, ymax])
            # boxes.append([0, 0, 10, 10])
            # obbPoints.append(instance_points)
            labels.append(modelIdx+1)


        
        num_objs = len(data)
        # print("---------")
        # print(num_objs)
        # print(torch.tensor(labels, dtype=torch.int64))
        # print("---------")
        # print("IMG name {}".format(img_path))
        # print("NUM = {}".format(num_objs))

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        mask_tensor = torch.as_tensor(target_mask, dtype=torch.uint8)*100
        # cv2.imwrite( "/workspace/data/Obb/weight_file2/img/"+img_num+".png", mask_tensor.numpy())
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)



        target = {}
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["masks"] = mask_tensor
        target["image_id"] = image_id
        return img_re, target


#   All bounding boxes should have positive height and width. Found invalid box [0.0, 3.3333332538604736, 0.0, 1065.0] for target at index 0.
    def __len__(self):
        return len(self.imgs)

    def get_traindata_list(self, dir_path):
        rgb_sub_dir = "RGB"
        depth_sub_dir = "depth"

        rgb_dir = os.path.join(dir_path, rgb_sub_dir)
        depth_dir = os.path.join(dir_path, depth_sub_dir)

        file_dir_list = glob.glob(os.path.join(rgb_dir, "*"))
        file_name_list=[]
        for file_dir in file_dir_list:
            file_name_list.append(file_dir.split(os.sep)[-1])
        # print(file_name_list)
        # file_name_list = (*file_dir_list).split(os.sep)
        # print(file_list)
        return file_name_list
    def getModelIdx(self, id):
        # model_dict = {"0": "Safety Hat", "1": "Paint 5G Bucket", "2": "Jigsaw", "3": "Safety Goggles", "4": "obj_000001"}
        model_dict = {"0": "Kettlebell", "1": "CUP", "2": "Pillow"}
        for k, v in model_dict.items():
            if id==v:
                return int(k)
        print("[Error] Model not found!!!!")
        return -1