from tensorboardX import SummaryWriter
from torchvision import models

from Config import LOG_DIR


class Logger:

    writer = SummaryWriter()
    correct = 0
    incorrect = 0
    def writeTB(self , name, val , iter):
        self.writer.add_scalar('data/'+name, val, iter)



    def writeIMG(self, img, iter , name):
        self.writer.add_image(name, img, iter)

    def accuracy(self):
        total =(self.incorrect+self.correct)
        if total>0:
            x = float(self.correct)/float(total)
        else:
            x = 0
        return x


    def writeAccuracy(self,iter,train):
        if train:
            self.writeTB('TrainAccuracy',self.accuracy(),iter)
        else:
            self.writeTB('TestAccuracy', self.accuracy(), iter)
    def Correct(self):
        self.correct= self.correct+1

    def InCorrect(self):
        self.incorrect= self.incorrect+1

    def AddGraph(self,net,dummy_input):
        with SummaryWriter(comment="autoEncoder 1.0 ") as w:
            w.add_graph(net, dummy_input)


    def close(self):
        self.writer.export_scalars_to_json(LOG_DIR + "all_scalars.json")
        self.writer.close()