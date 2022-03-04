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
        
        dir_path = glob.glob(os.path.join(path,'*'))[0]

        self.dir_path = dir_path
        self.transforms = transforms
        # load all image files, sorting them to
        # ensure that they are aligned
        self.imgs = list(sorted(os.listdir(os.path.join(dir_path, "RGB"))))
        self.masks = list(sorted(os.listdir(os.path.join(dir_path, "mask"))))

        json_dir = os.path.join(dir_path, "gt.json")
        self.json_object = json.load(open(json_dir))
        # print(self.json_object)
        # print("----------------------------------------------")
        # print("dir_path = {}".format(self.dir_path))
        # print("Img_path = {}".format(self.imgs))
        # print("Mask_path = {}".format(self.masks))
        # print("----------------------------------------------") 
        return

    def __getitem__(self, idx):
        print("----------------------------------------------")
        print("IDX {}".format(idx))
        # load images ad masks
        img_path = os.path.join(self.dir_path, "RGB", self.imgs[idx])
        mask_path = os.path.join(self.dir_path, "mask", self.masks[idx])

        print("Img_path = {}".format(img_path))
        print("Mask_path = {}".format(mask_path))
        print("----------------------------------------------")

        img = cv2.imread(img_path)
        # mask = Image.open(mask_path).convert("L")
        data = self.json_object[str(idx)]["Objects"]
        print(len(data))
        print(data)

        cvcv = cv2.imread(img_path)

        for instance_idx, instance_list in enumerate(data):
            instance = instance_list["projection point"]
            for points in instance:
                print(points)
                print(type(points))
                for k, v in points.items():
                    print("{}, {}".format(k, v))
                    cvcv = cv2.circle(cvcv, (int(v[0]), int(v[1])), 3, sampleColor.color_list[instance_idx], -1)
                
            print("  ---")
        cv2.imshow("dasf", cvcv)
        cv2.waitKey()
        target = "TEMP"
        return img, target

        mask = np.array(mask)
        # instances are encoded as different colors
        obj_ids = np.unique(mask)
        # first id is the background, so remove it
        obj_ids = obj_ids[2:]
 
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # print("------------- MASK ------------------")
        # print(mask)
        # print(mask.shape)
        # print(obj_ids)
        # print(masks)
        # get bounding box coordinates for each mask
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
 
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
 
        if self.transforms is not None:
            img, target = self.transforms(img, target)
 
        return img, target
 
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
        print(file_name_list)
        # file_name_list = (*file_dir_list).split(os.sep)
        # print(file_list)
        return file_name_list
    def getRGBDtensor(self, dir_path):
        return