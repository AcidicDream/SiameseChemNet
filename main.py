import cv2
import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
from torchvision.models import vgg16_bn

from Config import train_batch_size, train_number_epochs
from Eval import get_accuracy
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os

from nets.AutoEncoder import autoencoder
from nets.SiamNet import siamAutoencoder

from utils.Logger import Logger

from dataReader.DataReader import load_checkpoint, train_test_split, save_checkpoint

from utils.LossFunctions import ContrastiveLoss

from dataReader.Dataset import get_siam_set

from utils.Visualize import show_plot

currentEpoch=0

#validate using a test set
validate = True

#define a logger for tensorboard
log = Logger()


#define network to use
net = siamAutoencoder().cuda()
optimizer = optim.Adam(net.parameters(),lr = 0.0005 )
#net = load_checkpoint(net,optimizer)
criterion = ContrastiveLoss()


#net = autoencoder().cuda()
#criterion = nn.MSELoss()
#optimizer = torch.optim.Adam(net.parameters(), lr=0.0005,weight_decay=1e-5)


#todo modify logger
dummy_input = Variable(torch.rand(1, 1, 28, 28)).cuda()
log.AddGraph(net,(dummy_input,dummy_input))




#load the train and test list
list = pd.read_csv('/home/jasper/Documents/BP_Jasp/data/pgbalancedList.csv', sep=',', header=None)
trainIndx, testIndx, valIndx = train_test_split(list,splits=[0.7, 0.1, 0.2])
siamese_dataset = get_siam_set(list[0:trainIndx])
testSet = np.transpose( list[trainIndx:trainIndx+testIndx])
triple_list = np.empty_like(testSet)
triple_list[:]= testSet
train_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=100)




def run():
    counter = []
    loss_history = []
    iteration_number = 0
    for epoch in range(currentEpoch, currentEpoch + train_number_epochs):
        for i, data in enumerate(train_dataloader, 0):
            loss = net.trainBatch(data, optimizer, criterion)
            if i % 10 == 0:
                log.writeTB('loss', loss, iteration_number)
                if (validate):
                    accu = get_accuracy(net , 1.5, epoch)
                    log.writeTB('testAccuracy', accu, iteration_number)
                    print("Epoch number {}\n Current loss {}\n accuracy  {}\n ".format(epoch, loss, accu))
                else:
                    print("Epoch number {}\n Current loss {}\n  ".format(epoch, loss))
                iteration_number += 10
                counter.append(iteration_number)
                loss_history.append(loss)
    save_checkpoint(net, epoch, optimizer)
    log.close()
    show_plot(counter, loss_history)

run()