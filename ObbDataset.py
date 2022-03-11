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
        
        dir_path = glob.glob(os.path.join(path,"19201080_100_multi_OBB"))[0]
        dir_path = glob.glob(os.path.join(path,"640480_100_multi_OBB_v"))[0]

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
        # print("----------------------------------------------")
        # load images ad masks
        img_path = os.path.join(self.dir_path, "RGB", self.imgs[idx])
        img_num = img_path.split(os.sep)[-1][:-4]
        data = self.json_object[str(img_num)]["Objects"]
        img_re = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)

        # print("IDX {}".format(img_num))
        # print("Img_num = {}".format(img_num))
        # print("Img_path = {}".format(img_path))
        # print("Mask_path = {}".format(mask_path))
        # print("----------------------------------------------")

        obbPoints=[]
        labels = []

        # cvcv = cv2.imread(img_path)

        for instance in data:
            # print("\ninstance {}".format(instance))
            instance_points = list(instance["projection point"][0].values())
            instance_labels = instance["model"]
            modelIdx = self.getModelIdx(instance_labels)
            # print("Points {}".format(instance_points))
            # print("Labels {}".format(instance_labels))
            obbPoints.append(instance_points)
            labels.append(modelIdx)
        # print("--------------------------------------------------------")

            # print("Model num {}".format(len(instance_list["model"])))
            # for instance in instance_list:
            #     print(" Instance {}".format(instance))
                # target_points = list(instance["projection point"].values())
                # boxes.append(target_points)
            #     for k, v in points.items():
            #         print("{}, {}".format(k, v))
            #         cvcv = cv2.circle(cvcv, (int(v[0]), int(v[1])), 3, sampleColor.color_list[instance_idx], -1)
            # print("  ---")
        # cv2.imshow("dasf", cvcv)
        # cv2.waitKey()
        mask_path = os.path.join(self.dir_path, "mask", self.masks[idx])
        mask = cv2.imread(mask_path)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[2:]
        num_objs = len(obj_ids)

        boxes = []
        for i in range(num_objs):
            pos = np.where(mask[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            # boxes.append([xmin, ymin, xmax, ymax])
            boxes.append([0, 0, 10, 10])
 
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # there is only one class
        mask_tensor = torch.as_tensor(mask, dtype=torch.uint8)
 
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        # suppose all instances are not crowd
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)



        target = {}
        # target["obbPoints"] = obbPoints
        # target["labels"] = labels
        # target["idx"] = img_num
        # print("Labels = {}".format(labels))
        # print("Labels = {}".format(type(labels)))
        target["boxes"] = boxes
        target["labels"] = torch.tensor(labels, dtype=torch.int64)
        # target["masks"] = mask_tensor[:len(obj_ids)]
        target["masks"] = mask_tensor
        # target["image_id"] = image_id
        # target["area"] = area
        # target["iscrowd"] = iscrowd
        # print("OBB Point = {}".format(obbPoints))
        # print("labels = {}".format(labels))
        # print("labels = {}".format(target["labels"]))
        return img_re, target

        # instances are encoded as different colors
        # first id is the background, so remove it
 
        # split the color-encoded mask into a set
        # of binary masks
        masks = mask == obj_ids[:, None, None]
        # print("------------- MASK ------------------")
        # print(mask)
        # print(mask.shape)
        # print(obj_ids)
        # print(masks)
        # get bounding box coordinates for each mask
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
        print(file_name_list)
        # file_name_list = (*file_dir_list).split(os.sep)
        # print(file_list)
        return file_name_list
    def getModelIdx(self, id):
        model_dict = {"1": "Safety Hat", "2": "Paint 5G Bucket", "3": "Jigsaw", "4": "Safety Goggles", "5": "obj_000001"}
        for k, v in model_dict.items():
            if id==v:
                return int(k)
        print("[Error] Model not found!!!!")
        return -1