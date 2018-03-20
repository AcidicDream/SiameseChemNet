import shlex
import pandas as pd
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image, ImageChops
import torch
from torch.autograd import Variable
import PIL.ImageOps
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from Config import IMAGE_DIR, GrayScale
from utils.preProcess import preProcess


class SiameseNetworkDataset(Dataset):

    def __init__(self, triple_list, transform=None, should_invert=False):
        self.transform = transform
        self.should_invert = should_invert
        self.ref_list = triple_list[0]
        self.sim_list = triple_list[1]
        self.label_list = triple_list[2]

    def __getitem__(self, index):
        self.lastLabel = self.label_list[index]
        img0_tuple = self.ref_list[index]
        img1_tuple = self.sim_list[index]
        stable = Image.open('/home/jasper/Documents/BP_Jasp/project/util/stable.png' )
        img0 = Image.open(IMAGE_DIR +img0_tuple)
        img1 = Image.open(IMAGE_DIR +img1_tuple)
        diff = preProcess(img0,img1)
        if (GrayScale):
            diff= diff.convert("L")
            stable = stable.convert("L")

        img1 = ImageChops.subtract(img0, img1)
        if self.should_invert:
            stable = PIL.ImageOps.invert( stable)
            diff = PIL.ImageOps.invert( diff)

        if self.transform is not None:
            stable = self.transform(stable)
            diff = self.transform(diff)

        return stable, diff, torch.from_numpy(np.array([int(self.label_list[index])], dtype=np.float32))

    def __len__(self):
        return len(self.ref_list)



def get_siam_set(list):
    return SiameseNetworkDataset(triple_list=list,
                          transform=transforms.Compose([
                              transforms.Scale(30),
                              transforms.CenterCrop(28),
                              transforms.ToTensor()
                          ])
                          , should_invert=False)
