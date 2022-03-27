from train import Train
from inference  import Inference
from util.yolo_result import *


if __name__=="__main__":
    print("asasdffdfdf")
    # trainer = Train()
    # trainer.train()
    # return

    yolo = YOLOresult()
    path_list = yolo.get_image_list(path="/workspace/src/darknet/build/darknet/x64/data/train.txt")
    yolo.parsing("/workspace/src/darknet/result.txt", path_list)
    yolo.print_yoloresult()

    trainer = Train()
    trainer.train(yolo)

    infer = Inference()
    infer.inference("./weights", yolo)
    # print("ddd")
    # print(path)
    # dir_path = glob.glob(os.path.join(path,'*'))[0]
    # print(dir_path)


    # dataloader.getRGBDtensor(dir_path)
    # main()
