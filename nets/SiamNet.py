import torch
import numpy as np
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F

from dataReader.DataReader import load_checkpoint
from utils.Visualize import to_img, diff


class siamAutoencoder(nn.Module):
    def __init__(self):

        super(siamAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=1),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=2, padding=1),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.encoder = load_checkpoint(self.encoder)

        self.fc1 = nn.Sequential(
            nn.Linear(8 * 2 * 2, 16),
            nn.ReLU(inplace=True),

            nn.Linear(16, 8),
            nn.ReLU(inplace=True),

            nn.Linear(8, 1),
        )

    def forward_once(self, x):
        x = self.encoder(x)
        x = x.view(x.size()[0], -1)
        x = self.fc1(x)
        return x


    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


    def name(self):
        return "SiamNet 1.0"

    def dummyInput(self):
        return Variable(torch.rand(1, 8, 2, 2))

    def trainBatch(self, data, optimizer, criterion):
        img0, img1, label1 = data
        diff1 = img0 - img0
        diff2 = diff(np.asarray(img1), np.asarray(img0))
        diff1, diff2 = Variable(diff1).cuda(), Variable(torch.FloatTensor(diff2)).cuda()
        img0, img1, label = Variable(img0).cuda(), Variable(img1).cuda(), Variable(label1).cuda()
        output1, output2 = self(diff1, diff2)
        optimizer.zero_grad()
        loss_contrastive = criterion(output1, output2, label)
        loss_contrastive.backward()
        optimizer.step()
        return loss_contrastive.data[0]

    def Eval(self, data):
        img0, img1, label1 = data
        output1, output2 = self(Variable(img0).cuda(), Variable(img1).cuda())
        prediction = F.pairwise_distance(output1, output2).cpu().data.numpy()[0][0]
        return label1, prediction
