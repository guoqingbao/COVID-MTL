import math
import gc
import torch
from enum import Enum
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from .SqueezeNet3D import SqueezeNet3D
from torch.utils.data import DataLoader
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset as torchDataset
from sklearn.model_selection import train_test_split
from .utils import * 


class DataType(Enum):
    VOLUME_2 = 1,
    VOLUME_3 = 2,
    VOLUME_RADIO = 3


class CovidDataset(torchDataset):
    def __init__(self, datatype, datalst, labels, path, CTMEAN, CTSTD, augment = False):
        self.datatype = datatype
        if datatype == DataType.VOLUME_RADIO:
            self.data = datalst[0] # volume id (patient id for loading 3D segmented lung volume)
            self.r_data = datalst[1] # radio features

            # labels for three tasks
            self.labels_ct = labels[0] # ct label
            self.labels_nc = labels[1] # nat label
            self.labels_sv = labels[2] # severity label
        elif datatype == DataType.VOLUME_2 or datatype == DataType.VOLUME_3: # in case you want only volume inputs, and/or two tasks 
            self.data = datalst[0] # volume id
            self.labels_ct = labels[0] # ct label
            self.labels_nc = labels[1] # nat label
            if datatype == DataType.VOLUME_3:
                self.labels_sv = labels[2] # severity label
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
        # load volume data
        X =  GetVolume(pid, self.aug, self.path, self.CTMEAN, self.CTSTD)
        
        # label for the infection
        y_ct = list(self.labels_ct)[idx]
        y_nc = list(self.labels_nc)[idx]

        # another task label, or dual inputs
        if self.datatype == DataType.VOLUME_RADIO or self.datatype == DataType.VOLUME_3:
            y_sv = list(self.labels_sv)[idx]

        # dual inputs (volume + features)
        if self.datatype == DataType.VOLUME_RADIO:
            XR = self.r_data[idx]
            return X, XR, y_ct, y_nc, y_sv
        else:
            if self.datatype == DataType.VOLUME_3:
                return X, y_ct, y_nc, y_sv
            else:
                return X, y_ct, y_nc

# The FNN branch of COVID-MTL
class FNN(nn.Module):

    def __init__(self, input_size, output_size):
        super(FNN, self).__init__()
        
        self.features = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(inplace=False),
            
            nn.Linear(256, 128),
            nn.ReLU(inplace=False),
            
            nn.Linear(128, 64),
            nn.ReLU(inplace=False)
        )
        self.classifier = nn.Linear(64, output_size)

    def forward(self, x):
        x = self.features(x)
        out = self.classifier(x)
        return out

# the COVID-MTL model
class COVIDMTL(nn.Module):
    def __init__(self, datatype, radiofeatures = 375, shift3d = True, shift_chance=0.25, decay_iterations=0, num_classes=[2, 2, 3], sample_depth = DEPTH, sample_size = WIDTH):
        super(COVIDMTL,self).__init__()
        self.datatype = datatype
        last_duration = int(math.ceil(sample_depth / 32))
        last_size = int(math.ceil(sample_size / 32))

        # in case of dual inputs
        if datatype == DataType.VOLUME_RADIO:
            self.radioFNN = FNN(radiofeatures, 32)

        # backbone (shift3d indicate whether the backbone has a shift3d layer), pass use_classifier=False to extract shared features for all tasks
        self.features = SqueezeNet3D(shift3d=shift3d, shift_chance=shift_chance, decay_iterations=decay_iterations, use_classifier=False)
        
        # the following are task-specific layers
        self.classifier1_ = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Conv3d(512, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        )
        self.classifier2_ = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Conv3d(512, 32, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
        )

        if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
            self.classifier3_ = nn.Sequential(
                nn.Dropout(p=0.3),
                nn.Conv3d(512, 32, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.AvgPool3d((last_duration, last_size, last_size), stride=1)
            )

        if datatype == DataType.VOLUME_RADIO: # dual inputs with feature concatenation
            self.classifier1 = nn.Sequential(
                                nn.ReLU(inplace=False),
                                nn.Linear(32+32, num_classes[0])
                                )
            
            self.classifier2 = nn.Sequential(
                                nn.ReLU(inplace=False),
                                nn.Linear(32+32, num_classes[1])
                                )
            self.classifier3 = nn.Sequential(
                                nn.ReLU(inplace=False),
                                nn.Linear(32+32, num_classes[2])
                                )
        else:
            self.classifier1 = self.classifier1_
            self.classifier2 = self.classifier2_
            if datatype == DataType.VOLUME_3:
                self.classifier3 = self.classifier3_

        # model initialization
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    # first extract shared features with 3D CNN, then extract high-level radio features (if dual inputs)
    # and finally, task specific layers and decision layer
    def forward(self,x, x_radio):
        x = self.features(x)

        if self.datatype == DataType.VOLUME_RADIO:
            x_radio = self.radioFNN(x_radio)
            covid_ct = self.classifier1_(x)
            covid_ct = covid_ct.view(covid_ct.size(0), -1)
            
            covid_nc = self.classifier2_(x)
            covid_nc = covid_nc.view(covid_nc.size(0), -1)

            covid_sv = self.classifier3_(x)
            covid_sv = covid_sv.view(covid_sv.size(0), -1)
            
            # concatenate 3D CNN features with radio features
            covid_ct = torch.cat((covid_ct, x_radio), 1)
            covid_nc = torch.cat((covid_nc, x_radio), 1)
            covid_sv = torch.cat((covid_sv, x_radio), 1)

            # decision layer
            covid_ct = self.classifier1(covid_ct)
            covid_nc = self.classifier2(covid_nc)
            covid_sv = self.classifier3(covid_sv)

            return [covid_ct, covid_nc, covid_sv]
        else:
            covid_ct = self.classifier1(x)
            covid_ct = covid_ct.view(covid_ct.size(0), -1)
            
            covid_nc = self.classifier2(x)
            covid_nc = covid_nc.view(covid_nc.size(0), -1)

            if self.datatype == DataType.VOLUME_3:
                covid_sv = self.classifier3(x)
                covid_sv = covid_sv.view(covid_sv.size(0), -1)
                return [covid_ct, covid_nc, covid_sv]
            else:
                return [covid_ct, covid_nc]

# multitask loss function (random-weighted loss integrated)
class MultiTaskLoss(nn.Module):
    def __init__(self, datatype, random_weighted_loss = True, draw = 2):
        super(MultiTaskLoss, self).__init__()
        self.datatype = datatype
        self.draw = draw
        self.random_weighted_loss = random_weighted_loss
        # each task has its own loss
        self.loss_funcs = [nn.CrossEntropyLoss(), nn.CrossEntropyLoss()]
        if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
            self.loss_funcs.append(nn.CrossEntropyLoss())

        self.log_vars = nn.Parameter(torch.zeros((len(self.loss_funcs))))

    def forward(self, preds, labels):
        # calculate individual losses
        loss1 = self.loss_funcs[0](preds[0],labels[0])
        loss2 = self.loss_funcs[1](preds[1],labels[1])
        if self.datatype == DataType.VOLUME_RADIO or self.datatype == DataType.VOLUME_3:
            loss3 = self.loss_funcs[2](preds[2],labels[2])

        # loss combination with random-weighted loss
        if self.random_weighted_loss:
            if self.datatype == DataType.VOLUME_RADIO or self.datatype == DataType.VOLUME_3:
                pweights = np.sum(np.random.dirichlet(np.ones(3),size = self.draw), axis=0)
                return (loss1*pweights[0] + loss2*pweights[1] + loss3*pweights[2]) / self.draw
            else: # two-task can be either random-weighted or uniform
                proba = np.random.uniform(0,1)
                return loss1*proba + (1-proba)*loss2 
        else: # the competing loss (uncertainty loss)
            losses = torch.stack((loss1, loss2, loss3)).cuda()
            stds = (torch.exp(self.log_vars)**(1/2)).cuda()
            coeffs = 1 / (torch.Tensor([1, 1, 1]).cuda() * (stds**2))
            losses = coeffs*losses + torch.log(stds)
            return losses.mean()     


# test the model
def MTL_Test(model, loader, datatype, labels):
    model.eval()
    probas_ct_ = []
    probas_nat_ = []
    if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
        probas_sv_ = []

    if datatype == DataType.VOLUME_RADIO:
        for x1, x2, t_ct, t_nat, t_sv in loader:
            with torch.no_grad():
                v1, v2, t_ct, t_nat, t_sv = Variable(torch.HalfTensor(x1).cuda()),  Variable(torch.HalfTensor(x2).cuda()),   \
                     Variable(torch.LongTensor(t_ct.tolist()).cuda()),  Variable(torch.LongTensor(t_nat.tolist()).cuda()),    \
                         Variable(torch.LongTensor(t_sv.tolist()).cuda())
                y = model(v1, v2)
                probas_ct_.extend(F.softmax(y[0]).cpu().numpy().tolist())
                probas_nat_.extend(F.softmax(y[1]).cpu().numpy().tolist())
                probas_sv_.extend(F.softmax(y[2]).cpu().numpy().tolist())
    elif datatype == DataType.VOLUME_3:
        for x1, t_ct, t_nat, t_sv in loader:
            with torch.no_grad():
                v1, t_ct, t_nat, t_sv = Variable(torch.HalfTensor(x1).cuda()),  \
                     Variable(torch.LongTensor(t_ct.tolist()).cuda()),  Variable(torch.LongTensor(t_nat.tolist()).cuda()),    \
                         Variable(torch.LongTensor(t_sv.tolist()).cuda())
                y = model(v1)
                probas_ct_.extend(F.softmax(y[0]).cpu().numpy().tolist())
                probas_nat_.extend(F.softmax(y[1]).cpu().numpy().tolist())
                probas_sv_.extend(F.softmax(y[2]).cpu().numpy().tolist())    
    else:
        for x1, t_ct, t_nat in loader:
            with torch.no_grad():
                v1, t_ct, t_nat = Variable(torch.HalfTensor(x1).cuda()),  \
                     Variable(torch.LongTensor(t_ct.tolist()).cuda()),  Variable(torch.LongTensor(t_nat.tolist()).cuda())
                y = model(v1)
                probas_ct_.extend(F.softmax(y[0]).cpu().numpy().tolist())
                probas_nat_.extend(F.softmax(y[1]).cpu().numpy().tolist())           

    probas_ct_ = np.array(probas_ct_)
    probas_nat_ = np.array(probas_nat_)

    pred_ct = np.argmax(probas_ct_,axis=1) 
    pred_nat = np.argmax(probas_nat_,axis=1) 


    ac = accuracy_score(labels[0], pred_ct)
    print("Testing accuracy (CT) {}\r\n".format(ac))

    ac = accuracy_score(labels[1], pred_nat)
    print("Testing accuracy (NAT) {}\r\n".format(ac))
    
    if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
        probas_sv_ = np.array(probas_sv_)
        pred_sv = np.argmax(probas_sv_,axis=1) 
        ac = accuracy_score(labels[2], pred_sv)
        print("Testing accuracy (Severity) {}\r\n".format(ac))
        return pred_ct, pred_nat, pred_sv, probas_ct_, probas_nat_, probas_sv_
    else:
        return pred_ct, pred_nat, probas_ct_, probas_nat_


class History:
    def __init__(self, datatype = DataType.VOLUME_RADIO):
        if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
            self.history = {'epoch':[], 'train_acc_ct':[], 'train_acc_nat':[], 'train_acc_severity':[], 'test_acc_ct':[], 'test_acc_nat':[], 'test_acc_severity':[], 'train_loss':[], 'test_loss':[]}
        else:
            self.history = {'epoch':[], 'train_acc_ct':[], 'train_acc_nat':[], 'test_acc_ct':[], 'test_acc_nat':[], 'train_loss':[], 'test_loss':[]}
    
    def add(self, e, train_acc_ct, train_acc_nat, test_acc_ct, test_acc_nat, train_loss, test_loss):
        self.history['epoch'].append(e)
        self.history['train_acc_ct'].append(train_acc_ct)
        self.history['train_acc_nat'].append(train_acc_nat)
        
        self.history['test_acc_ct'].append(test_acc_ct)      
        self.history['test_acc_nat'].append(test_acc_nat)        
        
        self.history['train_loss'].append(train_loss)        
        self.history['test_loss'].append(test_loss) 

    def add(self, e, train_acc_ct, train_acc_nat, train_acc_sv, test_acc_ct, test_acc_nat, test_acc_sv, train_loss, test_loss):
        self.history['epoch'].append(e)
        self.history['train_acc_ct'].append(train_acc_ct)
        self.history['train_acc_nat'].append(train_acc_nat)
        self.history['train_acc_severity'].append(train_acc_sv)   
        
        self.history['test_acc_ct'].append(test_acc_ct)      
        self.history['test_acc_nat'].append(test_acc_nat)        
        self.history['test_acc_severity'].append(test_acc_sv) 
        
        self.history['train_loss'].append(train_loss)        
        self.history['test_loss'].append(test_loss)  

# train and evaluate the model
def Run(model, trainLoader, testLoader, datatype, random_weighted_loss, epochs, base_lr, weight_decay, log_path, log_file):
    model = nn.DataParallel(model).cuda()
    cudnn.benckmark = True

    opt = optim.SGD(model.parameters(),
                    lr=base_lr,
                    momentum=0.9,
                    weight_decay=weight_decay,
                    nesterov=True)
#     loss_func = nn.CrossEntropyLoss().cuda()
    loss_func = MultiTaskLoss(datatype, random_weighted_loss = random_weighted_loss).cuda()

    if datatype == DataType.VOLUME_RADIO or DataType.VOLUME_3:
        if testLoader != None:
            headers = ["Epoch", "LR", "TrainLoss", "TestLoss", "TrainAcc(CT)", "TrainAcc(NAT)", "TrainAcc(Severity)", "TestAcc(CT)", "TestAcc(NAT)", "TestAcc(Severity)"]
        else:
            headers = ["Epoch", "LR", "TrainLoss", "TrainAcc(CT)", "TrainAcc(NAT)", "TrainAcc(Severity)"]
    else:
        if testLoader != None:
            headers = ["Epoch", "LR", "TrainLoss", "TestLoss", "TrainAcc(CT)", "TrainAcc(NAT)", "TestAcc(CT)", "TestAcc(NAT)"]
        else:
            headers = ["Epoch", "LR", "TrainLoss", "TrainAcc(CT)", "TrainAcc(NAT)"]

    logger = Logger(log_path, log_file, headers)  
    history = History()
    for e in range(epochs):
        lr = cosine_lr(opt, base_lr, e, epochs)
        model.train()
        train_loss, train_acc_ct, train_acc_nat, train_acc_sv, train_n = 0, 0, 0, 0, 0
        bar = tqdm(total=len(trainLoader), leave=False)
        for inputs in trainLoader:
            if datatype == DataType.VOLUME_RADIO:
                x1, x2, t_ct, t_nat, t_sv = inputs
                v1, v2, t_ct, t_nat, t_sv = Variable(torch.HalfTensor(x1).cuda()), Variable(torch.HalfTensor(x2).cuda()), Variable(torch.LongTensor(t_ct).cuda()), Variable(torch.LongTensor(t_nat).cuda()),  \
                     Variable(torch.LongTensor(t_sv).cuda()) 
                y = model(v1, v2)
                loss = loss_func(y, t_ct, t_nat, t_sv)
            elif datatype == DataType.VOLUME_3:
                x1, t_ct, t_nat, t_sv = inputs
                v1, t_ct, t_nat, t_sv = Variable(torch.HalfTensor(x1).cuda()), Variable(torch.LongTensor(t_ct).cuda()), Variable(torch.LongTensor(t_nat).cuda()),  \
                     Variable(torch.LongTensor(t_sv).cuda()) 
                y = model(v1)
                loss = loss_func(y, t_ct, t_nat, t_sv)
            else:
                x1, t_ct, t_nat = inputs
                v1, t_ct, t_nat = Variable(torch.HalfTensor(x1).cuda()), Variable(torch.LongTensor(t_ct).cuda()), Variable(torch.LongTensor(t_nat).cuda())
                y = model(v1)
                loss = loss_func(y, t_ct, t_nat)

            opt.zero_grad()
#             with torch.autograd.detect_anomaly():
            loss.backward()
            opt.step()

            train_acc_ct += accuracy(y[0], t_ct).item()
            train_acc_nat += accuracy(y[1], t_nat).item()
                                    
            train_loss += loss.item() * t_ct.size(0)
            train_n += t_ct.size(0)

            if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
                train_acc_sv += accuracy(y[2], t_sv).item()
                bar.set_description("Total Loss: {:.4f}, Accuracy(CT): {:.2f}, Accuracy(NAT): {:.2f}, Accuracy(Severity): {:.2f}".format(
                train_loss / train_n, train_acc_ct / train_n * 100, train_acc_nat / train_n * 100, train_acc_sv / train_n * 100), refresh=True)
            else:
                bar.set_description("Total Loss: {:.4f}, Accuracy(CT): {:.2f}, Accuracy(NAT): {:.2f}".format(
                train_loss / train_n, train_acc_ct / train_n * 100, train_acc_nat / train_n * 100), refresh=True)

            bar.update()
        bar.close()

        if testLoader!=None:
            model.eval()
            test_loss, test_acc_ct, test_acc_nat, test_acc_sv, test_n = 0, 0, 0, 0, 0
            bar = tqdm(total=len(testLoader), leave=False)
            with torch.no_grad():
                for inputs in testLoader:
                    if datatype == DataType.VOLUME_RADIO:
                        x1, x2, t_ct, t_nat, t_sv = inputs
                        v1, v2, t_ct, t_nat, t_sv = Variable(torch.HalfTensor(x1).cuda()), Variable(torch.HalfTensor(x2).cuda()), Variable(torch.LongTensor(t_ct).cuda()), Variable(torch.LongTensor(t_nat).cuda()),  \
                            Variable(torch.LongTensor(t_sv).cuda()) 
                        y = model(v1, v2)
                        loss = loss_func(y, t_ct, t_nat, t_sv)
                    elif datatype == DataType.VOLUME_3:
                        x1, t_ct, t_nat, t_sv = inputs
                        v1, t_ct, t_nat, t_sv = Variable(torch.HalfTensor(x1).cuda()), Variable(torch.LongTensor(t_ct).cuda()), Variable(torch.LongTensor(t_nat).cuda()),  \
                            Variable(torch.LongTensor(t_sv).cuda()) 
                        y = model(v1)
                        loss = loss_func(y, t_ct, t_nat, t_sv)
                    else:
                        x1, t_ct, t_nat = inputs
                        v1, t_ct, t_nat = Variable(torch.HalfTensor(x1).cuda()), Variable(torch.LongTensor(t_ct).cuda()), Variable(torch.LongTensor(t_nat).cuda())
                        y = model(v1)
                        loss = loss_func(y, t_ct, t_nat)

                    test_loss += loss.item() * t_ct.size(0)
                                            
                    test_acc_ct += accuracy(y[0], t_ct).item()
                    test_acc_nat += accuracy(y[1], t_nat).item()

                    if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
                        test_acc_sv += accuracy(y[2], t_sv).item()
                                            
                    test_n += t_ct.size(0)
                    bar.update()
            bar.close()
            if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
                logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                            train_acc_ct / train_n * 100, train_acc_nat / train_n * 100, train_acc_sv / train_n * 100,
                            test_acc_ct / test_n * 100, test_acc_nat / test_n * 100, test_acc_sv / test_n * 100) 
                            
                history.add(e+1, train_acc_ct / train_n * 100, train_acc_nat / train_n * 100, 
                        train_acc_sv / train_n * 100, test_acc_ct / test_n * 100, 
                        test_acc_nat / test_n * 100, test_acc_sv / test_n * 100, train_loss / train_n, test_loss / test_n)
            else:
                logger.write(e+1, lr, train_loss / train_n, test_loss / test_n,
                            train_acc_ct / train_n * 100, train_acc_nat / train_n * 100,
                            test_acc_ct / test_n * 100, test_acc_nat / test_n * 100) 

                history.add(e+1, train_acc_ct / train_n * 100, train_acc_nat / train_n * 100, test_acc_ct / test_n * 100, 
                        test_acc_nat / test_n * 100, train_loss / train_n, test_loss / test_n)
            
        else:
            if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
                logger.write(e+1, np.round(lr,5), train_loss / train_n, 
                            train_acc_ct / train_n * 100, train_acc_nat / train_n * 100, train_acc_sv / train_n * 100) 
                print("Epoch:", e+1, "\tLR:", np.round(lr,5), "\tTotal Train Loss:", np.round(train_loss / train_n,4), 
                    "\tTrain Acc (CT)", np.round(train_acc_ct / train_n * 100,4), "\tTrain Acc (NAT)", np.round(train_acc_nat / train_n * 100,4),
                    "\tTrain Acc (Severity)", np.round(train_acc_sv / train_n * 100,4))
                history.add(e+1, train_acc_ct / train_n * 100, train_acc_nat/train_n * 100,train_acc_sv/train_n * 100, -1, -1, -1, train_loss / train_n, -1)
            else:
                logger.write(e+1, np.round(lr,5), train_loss / train_n, 
                            train_acc_ct / train_n * 100, train_acc_nat / train_n * 100) 
                print("Epoch:", e+1, "\tLR:", np.round(lr,5), "\tTotal Train Loss:", np.round(train_loss / train_n,4), 
                    "\tTrain Acc (CT)", np.round(train_acc_ct / train_n * 100,4), "\tTrain Acc (NAT)", np.round(train_acc_nat / train_n * 100,4))
                history.add(e+1, train_acc_ct / train_n * 100, train_acc_nat/train_n * 100, -1, -1, train_loss / train_n, -1)        
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
    num_dataloader_workers = 25

    basicInfo = pd.read_pickle(path + 'data/info.ppl') # ppl file was saved under Python 3.8
    CTMEAN = np.round(np.mean(basicInfo.MeanValue),2)
    CTSTD = np.round(np.mean(basicInfo.StdValue),2)        
    morbidityClasses = [GetMorbidityClass(morbidity) for morbidity in basicInfo.Morbidity]
    basicInfo.loc[:, 'COVIDSeverity'] = morbidityClasses

    # radioinfo.ppl can be genrated by merging Patients.xlsx and radiofeatures.csv
    radiofeatures = pd.read_pickle(path + 'data/radioinfo.ppl') 
    radiofeatures.loc[:, 'COVIDNC'] = list(basicInfo.COVIDNC)
    radiofeatures.loc[:, 'COVIDCT'] = list(basicInfo.COVIDCT)
    radiofeatures.loc[:, 'COVIDSeverity'] = list(basicInfo.COVIDSeverity)

    x_train, x_test, y_train_nat, y_test_nat = train_test_split(basicInfo.PID, basicInfo.COVIDNC,stratify=basicInfo.COVIDNC, test_size=0.30, random_state=11, shuffle=True)

    y_train_ct = basicInfo.loc[y_train_nat.index].COVIDCT
    y_test_ct = basicInfo.loc[y_test_nat.index].COVIDCT

    y_train_sv = basicInfo.loc[y_train_nat.index].COVIDSeverity
    y_test_sv = basicInfo.loc[y_test_nat.index].COVIDSeverity


    scaler = MinMaxScaler()
    scaler.fit(radiofeatures.iloc[:,1:-3])
    scaledfeatures = pd.DataFrame(scaler.transform(radiofeatures.iloc[:,1:-3]),columns = radiofeatures.columns[1:-3])
    scaledfeatures.loc[:, 'PID'] = list(radiofeatures.PID)
    scaledfeatures.loc[:, 'COVIDNC'] = list(radiofeatures.COVIDNC)
    scaledfeatures.loc[:, 'COVIDCT'] = list(radiofeatures.COVIDCT)
    scaledfeatures.loc[:, 'COVIDSeverity'] = list(radiofeatures.COVIDSeverity)
    r_train = scaledfeatures.iloc[x_train.index,:-4]
    r_test = scaledfeatures.iloc[x_test.index,:-4]

    for datatype in [DataType.VOLUME_RADIO, DataType.VOLUME_3, DataType.VOLUME_2]:
        datalst = [x_train]
        if datatype == DataType.VOLUME_RADIO:
            datalst.append(r_train.values.astype(np.float16))
        labels = [y_train_ct, y_train_nat]
        if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
            labels.append(y_train_sv)
        trainDataset = CovidDataset(datatype, datalst, labels, path, CTMEAN, CTSTD, augment=True)
        trainLoader = DataLoader(trainDataset, batch_size=batch_size, num_workers=num_dataloader_workers)

        datalst1 = [x_test]
        if datatype == DataType.VOLUME_RADIO:
            datalst1.append(r_test.values.astype(np.float16))
        labels1 = [y_test_ct, y_test_nat]
        if datatype == DataType.VOLUME_RADIO or datatype == DataType.VOLUME_3:
            labels1.append(y_test_sv)

        testDataset = CovidDataset(datalst1, labels1, path, CTMEAN, CTSTD, augment=False)
        testLoader = DataLoader(testDataset, batch_size=batch_size, num_workers=num_dataloader_workers)
        logpath = path + 'log/mtl/' + str(datatype) + '/'
        for shif3d, randloss in zip([True, True, False, False], [True, False, True, False]):

            model = COVIDMTL(datatype, radiofeatures = len(r_train.columns), shift3d = shif3d).half()

            print('Number of model parameters: {}'.format(
                    sum([p.data.nelement() for p in model.parameters()])))
            logfile = str(shif3d) + '_' + str(randloss) + '.log'
            history = Run(model, trainLoader, testLoader, datatype, random_weighted_loss = randloss, epochs=80, base_lr=0.005, weight_decay=0.00005, log_path=logpath, log_file=logfile)
            torch.cuda.empty_cache()