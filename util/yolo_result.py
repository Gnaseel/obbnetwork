import numpy as np
import os

class Detection_item:
    def __init__(self):
        self.class_num=-1
        self.bbox = [0,0,0,0]
        return
    def __str__(self):
        return "    Class {} , Bbox = {} {} {} {}".format(self.class_num, *self.bbox)
class YOLOimage:
    def __init__(self):
        self.fileDir=""     # /workspace/src/darknet/build/darknet/x64/data/obj/4.jpg
        self.fileName=""    # 4
        self.item=[]
        return

    def get_bbox_point(self, text):
        # print("         BBBOX POINT = {}".format(text))
        left_idx = text.find("left_x")
        top_idx = text.find("top_y")
        width_idx = text.find("width")
        height_idx = text.find("height")
        left = int(text[left_idx+7:top_idx])
        top = int(text[top_idx+6:width_idx])
        width = int(text[width_idx+6:height_idx])
        height = int(text[height_idx+7:-1])
        # print("     {} {} {} {}".format(left, top, width, height))
        return [left, top, width, height]

    def get_item_data(self, text, classes):
        for idx, clss in enumerate(classes):
            i = text.find(clss)
            if i != -1:
                item = Detection_item()
                item.class_num = idx
                item.bbox = self.get_bbox_point(text)
                self.item.append(item)

class YOLOresult:
    def __init__(self):
        self.images=[]
        self.classes=["Kettlebell", "CUP", "Pillow"]
        return

    def get_image_list(self, path="/workspace/src/darknet/build/darknet/x64/data/train.txt"):
        f = open(path)
        lines = f.readlines()
        # print(lines)
        for idx, line in enumerate(lines):
            lines[idx] = self.delete_LF(line)
        return lines
    def print_yoloresult(self):

        sum_1 = []
        sum_2 = []
        count = 0
        for image in self.images:
            print("Path = {}".format(image.fileDir))
            for item in image.item:
                print("     Class = {},  bbox {} {} {} {}".format(self.classes[item.class_num], *item.bbox))
                sum_1.append(item.bbox[2])
                sum_2.append(item.bbox[3])
                count +=1
        sum_1 = np.array(sum_1)
        sum_2 = np.array(sum_2)
        print("{} {}".format(np.mean(sum_1), np.mean(sum_2)))
        print("{} {}".format(np.var(sum_1), np.var(sum_2)))
        print("{} {}".format(np.std(sum_1), np.std(sum_2)))
    def parsing(self, result_path, img_path_list):
        
        result = open(result_path)
        result_lines = result.readlines()
        result_idx = 0
        for path in img_path_list:

            # for idx, line in enumerate(result_lines[result_idx:]):
            while result_idx < len(result_lines):
                line = self.delete_LF(result_lines[result_idx])
                i = line.find(path)
                if i == -1:
                    result_idx +=1
                    continue
                # print("IDX {}".format(result_idx))
                # print(" DATA {}".format(line))
                
                yolo_image = YOLOimage()
                yolo_image.fileDir=path
                yolo_image.fileName=int(path.split(os.sep)[-1][:-4])
                
                result_idx +=1

                while result_idx < len(result_lines):
                    line = self.delete_LF(result_lines[result_idx])
                    i = line.find("/work")
                    if i != -1:
                        break
                    yolo_image.get_item_data(line, self.classes)
                    result_idx +=1
                self.images.append(yolo_image)
                result_idx -= 1
                break
                
        return

    def delete_LF(self, string):
        if string[-1]=="\n":
            string=string[:-1]
        return string

    def normal_obbpoint(self, target, bbox):

        # print(target)
        # print(bbox)
        ratioX = 120/(bbox[1] - bbox[0])
        ratioY = 120/(bbox[3] - bbox[2])
        point_list = np.array(target)

        point_list[0::2] -=(bbox[1] + bbox[0])/2
        point_list[1::2] -=(bbox[3] + bbox[2])/2
        point_list[0::2] *=ratioX
        point_list[1::2] *=ratioY


        # point_list[:,0] -=(bbox[1] + bbox[0])/2
        # point_list[:,1] -=(bbox[3] + bbox[2])/2
        # point_list[:,0] *=ratioX
        # point_list[:,1] *=ratioY
        # print(point_list.shape)
        # print(ratioX)
        # print(ratioY)
        return point_list
    
    def unnormal_obbpoint(self, target, bbox):
        ratioX = 120/(bbox[1] - bbox[0])
        ratioY = 120/(bbox[3] - bbox[2])
        point_list = np.array(target)

        point_list[0::2] /=ratioX
        point_list[1::2] /=ratioY
        point_list[0::2] +=(bbox[1] + bbox[0])/2
        point_list[1::2] +=(bbox[3] + bbox[2])/2

        # point_list = torch.Tensor(under_points)
        # print(point_list.shape)
        # print(ratioX)
        # print(ratioY)
        return point_list