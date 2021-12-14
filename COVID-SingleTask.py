import os
import gc
import pickle
import pandas as pd
import numpy as np
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.metrics import accuracy_score, auc
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as torchDataset
from .SqueezeNet3D import SqueezeNet3D
from .ResNet3D import resnet3d34
from .utils import *

class CovidDataset(torchDataset):
    def __init__(self, data, labels, path, CTMEAN, CTSTD, augment = False):
        self.labels = labels
        self.data = data
        self.aug = augment
        self.path = path
        self.CTMEAN = CTMEAN
        self.CTSTD = CTSTD

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
            
        pid = list(self.data)[idx]
        X =  GetVolume(pid, self.aug, self.path, self.CTMEAN, self.CTSTD)
        
        y = list(self.labels)[idx]
        
        return X, y

def Test(model, loader, y_test):
    model.eval()
    probas_ = []
    for x1, t in loader:
        with torch.no_grad():
            v, t = Variable(torch.HalfTensor(x1).cuda()), \
                            Variable(torch.LongTensor(t.tolist()).cuda())

            y = model(v)
            probas_.extend(F.softmax(y).cpu().numpy().tolist())
            
    probas_ = np.array(probas_)

    pred = np.argmax(probas_,axis=1) 

    ac = accuracy_score(y_test, pred)
    print("Testing accuracy {}\r\n".format(ac))
    
    return pred, probas_

class History:
    def __init__(self):
        self.history = {'epoch':[], 'train_acc':[], 'test_acc':[], 'train_loss':[], 'test_loss':[]}
    def add(self, e, train_acc, test_acc, train_loss, test_loss):
        self.history['epoch'].append(e)
        self.history['train_acc'].append(train_acc)
        self.history['test_acc'].append(test_acc)      
        
        self.history['train_loss'].append(train_loss)        
        self.history['test_loss'].append(test_loss)  
        
def Run(model, trainLoader, testLoader, epochs, base_lr, weight_decay, log_path, log_file):
    model = nn.DataParallel(model).cuda()
    cudnn.benckmark = True

    opt = optim.SGD(model.parameters(),
                    lr=base_lr,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    nesterov=True)
    loss_func = nn.CrossEntropyLoss().cuda()

    if testLoader != None:
        headers = ["Epoch", "LR", "TrainLoss", "TestLoss", "TrainAcc", "TestAcc"]
    else:
        headers = ["Epoch", "LR", "TrainLoss", "TrainAcc"]

    logger = Logger(log_path, log_file, headers)  
    history = History()
    for e in range(epochs):
        lr = cosine_lr(opt, base_lr, e, epochs)
        model.train()
        train_loss, train_acc, train_n = 0, 0, 0
        bar = tqdm(total=len(trainLoader), leave=False)
        for x1, t in trainLoader:
            v, t = Variable(torch.HalfTensor(x1).cuda()), \
                           Variable(torch.LongTensor(t).cuda())
            y = model(v)
            loss = loss_func(y, t)
            opt.zero_grad()
#             with torch.autograd.detect_anomaly():
            loss.backward()
            opt.step()

            train_acc += accuracy(y, t).item()
                                    
            train_loss += loss.item() * t.size(0)
            train_n += t.size(0)

            bar.set_description("Total Loss: {:.4f}, Accuracy: {:.2f}".format(
            train_loss / train_n, train_acc/train_n * 100), refresh=True)
            bar.update()
        bar.close()

        if testLoader!=None:
            model.eval()
            test_loss, test_acc, test_n = 0, 0, 0
            bar = tqdm(total=len(testLoader), leave=False)
            with torch.no_grad():
                for x1, t in testLoader:
                    v, t = Variable(torch.HalfTensor(x1).cuda()), \
                                   Variable(torch.LongTensor(t).cuda())
                    y = model(v)   
                    loss = loss_func(y, t)
                    test_loss += loss.item() * t.size(0)
                                            
                    test_acc += accuracy(y, t).item()
                                            
                    test_n += t.size(0)
                    bar.update()
            bar.close()
            
            logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                        train_acc / train_n * 100, test_acc / test_n * 100) 

            history.add(e+1, train_acc / train_n * 100, test_acc / test_n * 100, train_loss / train_n, test_loss / test_n)            
        else:
            logger.write(e+1, np.round(lr,5), train_loss / train_n, train_acc / train_n * 100) 
            print("Epoch:", e+1, "\tLR:", np.round(lr,5), "\tTrain Loss:", np.round(train_loss / train_n,4), 
                  "\tTrain Acc", np.round(train_acc / train_n * 100,4))
            
            history.add(e+1, train_acc / train_n * 100, -1, train_loss / train_n, -1)
    gc.collect()

    return history

def GetMorbidityClass(morbidity):
    if morbidity == 'Control' or morbidity == 'Suspected':
        return 0
    elif morbidity == 'Mild' or morbidity == 'Regular':
        return 1
    elif morbidity == 'Severe' or morbidity == 'Critically ill':
        return 2
    else:
        assert(False)

if __name__ == "__main__":
    path = '/guoqing/project/covid/'
    batch_size = 10
    epochs = 80
    num_dataloader_workers = 16
    basicInfo = pd.read_pickle(path + 'info.ppl')
    CTMEAN = np.round(np.mean(basicInfo.MeanValue),2)
    CTSTD = np.round(np.mean(basicInfo.StdValue),2)        
    morbidityClasses = [GetMorbidityClass(morbidity) for morbidity in basicInfo.Morbidity]
    basicInfo.loc[:, 'COVIDSeverity'] = morbidityClasses

    x_train, x_test, y_train_nat, y_test_nat = train_test_split(basicInfo.PID, basicInfo.COVIDNC,stratify=basicInfo.COVIDNC, test_size=0.30, random_state=11, shuffle=True)

    y_train_ct = basicInfo.loc[y_train_nat.index].COVIDCT
    y_test_ct = basicInfo.loc[y_test_nat.index].COVIDCT

    y_train_sv = basicInfo.loc[y_train_nat.index].COVIDSeverity
    y_test_sv = basicInfo.loc[y_test_nat.index].COVIDSeverity

    for shift3d, modelname in zip([True, False, False, False], ['shiftnet3d', 'sequeezenet3d', 'decovnet3d', 'resnet3d']):
        logpath = path + '/log/single/'
        logfile = str(shift3d) + '_'+modelname+'.log'

        classesnames = ['Negative', 'Positive']
        colors = ['blue', 'yellow', 'green']

        for targets, name in zip([(y_train_ct, y_test_ct), (y_train_nat, y_test_nat), (y_train_sv, y_test_sv)], ['ct', 'nat', 'sv']):
                
            trainDataset = CovidDataset(x_train, targets[0], path, CTMEAN, CTSTD, augment=True)
            trainLoader = DataLoader(trainDataset, batch_size=batch_size, num_workers=num_dataloader_workers)

            testDataset = CovidDataset(x_test, targets[1], path, CTMEAN, CTSTD, augment=False)
            testLoader = DataLoader(testDataset, batch_size=batch_size, num_workers=num_dataloader_workers)

            if modelname == 'resnet3d':
                model = resnet3d34(shift3d = shift3d, num_classes=3 if name == 'sv' else 2, shift3d=True).half()
            elif modelname == 'decovnet3d':
                #ENModel can be found in decovnet: https://github.com/sydney0zq/covid-19-detection/tree/master/deCoVnet/model
                # note: ENModel (DecovNet) need to be revised to accept shift3d layer (if you want to test its performance under shift3d)
                model = ENModel(shift3d = shift3d, crop_h=HEIGHT, crop_w=WIDTH, num_frames=DEPTH, num_classes=3 if name == 'sv' else 2).half()
            else:
                model = SqueezeNet3D(shift3d = shift3d, num_classes=3 if name == 'sv' else 2).half()


            print('Number of model parameters: {}'.format(
                    sum([p.data.nelement() for p in model.parameters()])))

            history = Run(model, trainLoader, testLoader, epochs=epochs, base_lr=0.005, weight_decay=0.00005, log_path=logpath, log_file=logfile)
 
            torch.cuda.empty_cache()
    