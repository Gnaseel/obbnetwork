from unittest import loader

import glob
import os
from train import Train



if __name__=="__main__":
    trainer = Train()
    trainer.train()
    # print(path)
    # dir_path = glob.glob(os.path.join(path,'*'))[0]
    # print(dir_path)


    # dataloader.getRGBDtensor(dir_path)
    # main()
