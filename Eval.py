import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
from torchvision.models import vgg16_bn

from dataReader.DataReader import get_triples, save_checkpoint, load_checkpoint
from dataReader.Relabeler import write_list, train_test_split, update_list
from logger import Logger
from nets.SiamNetwork import ContrastiveLoss, SiameseNetwork, SiameseNetworkRGB, SiameseNetwork2
from dataReader.dataset import SiameseNetworkDataset, get_siam_set
from nets.alexnet import get_alexSet
from nets.autoEncoder import siamAutoencoder
from nets.resnet import ResNet18, ResNetSiam
from util.config import DATA_DIR, IMAGE_DIR, train_batch_size, NUMBER_OF_EPOCH
from util.plot_image import show_plot, imshow
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import pandas as pd
import torch.nn as nn
from torch import optim
import torchvision.utils as vutils
import torch.nn.functional as F
import os

from utils.Visualize import plot_data


def get_accuracy(net , test_dataloader, threshHold, epoch):
    dataiter = iter(test_dataloader)
    all = []
    all_labels = []
    correct = 0
    incorrect = 0
    for i in range(100):
        data = next(dataiter)
        label, prediction = net.Eval(net,data)
        all.extend(prediction.tolist())
        all_labels.extend(label.data.cpu().numpy().tolist())
        if (label > 0.5).numpy() & (prediction < threshHold):
            correct = correct+1
        elif (label < 0.5).numpy() & (prediction > threshHold):
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    numpy_all = np.array(all)
    numpy_labels = np.array(all_labels)
    plot_data(numpy_all, numpy_labels,epoch)
    return correct/(correct+incorrect)