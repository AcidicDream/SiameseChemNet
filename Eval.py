import numpy as np
from torch.utils.data import DataLoader

from dataReader.DataReader import train_test_split
from dataReader.Dataset import get_siam_set
from utils.Visualize import plot_data
import pandas as pd


#load the train and test list
list = pd.read_csv('/home/jasper/Documents/BP_Jasp/data/pgbalancedList.csv', sep=',', header=None)
trainIndx, testIndx, valIndx = train_test_split(list,splits=[0.7, 0.1, 0.2])
testSet = np.transpose( list[trainIndx:trainIndx+testIndx])
triple_list = np.empty_like(testSet)
triple_list[:]= testSet
test_dataset = get_siam_set(triple_list)
test_dataloader = DataLoader(test_dataset, num_workers=1, batch_size=1, shuffle=False)


def get_accuracy(net , threshHold, epoch):
    dataiter = iter(test_dataloader)
    all = []
    all_labels = []
    correct = 0
    incorrect = 0
    for i in range(100):
        dat = next(dataiter)
        label, prediction = net.Eval(dat)
        label = dat[2][0][0]
        all.append(prediction)
        all_labels.append(label)
        if (label > 0.5) & (prediction < threshHold):
            correct = correct+1
        elif (label < 0.5) & (prediction > threshHold):
            correct = correct + 1
        else:
            incorrect = incorrect + 1
    numpy_all = np.array(all)
    numpy_labels = np.array(all_labels)
    plot_data(numpy_all, numpy_labels,epoch)
    return correct/(correct+incorrect)