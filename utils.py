
import cv2
import io
import math
import numpy as np
import pydicom
import random
import cv2
import os
import warnings;
warnings.filterwarnings('ignore');
import matplotlib.pyplot as plt
import pandas as pd
import datetime
from pydicom.encaps import decode_data_sequence 
import albumentations as A
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
from sklearn.metrics import accuracy_score, auc
DEPTH = 200
WIDTH = 250
HEIGHT = 250

def GetCropped(srcFile):
    dataset = pydicom.dcmread(srcFile)
    dataset.file_meta.TransferSyntaxUID
    dataset.BitsAllocated = 16
    frames = decode_data_sequence(dataset.PixelData)
    restored = []
    for f in frames:
        buf = io.BytesIO(f)
        img = cv2.imdecode(np.frombuffer(buf.getbuffer(), np.ushort), -1)
        restored.append(img)        
    restored = np.array(restored, dtype=np.ushort)
    return restored

noise_transform = A.Compose([
    A.IAAAdditiveGaussianNoise([10, 50], p=0.20)
])

def AugmentNorm(restored, aug, CTMEAN, CTSTD):
    if aug:
        restored = noise_transform(image=restored)['image']
    
    #norm
    restored = (restored - CTMEAN)/CTSTD
    
    # random rotation and flip
    if aug:
        r = random.randint(0,20)
        if r < 3:
            return np.rot90(restored, axes=(1,2), k=1).reshape((1, restored.shape[0], restored.shape[1],restored.shape[2])).astype(np.float16)
        elif r == 10 or r == 20:
            return np.rot90(restored, axes=(1,2), k=2).reshape((1, restored.shape[0], restored.shape[1],restored.shape[2])).astype(np.float16)
        elif r == 15:
            return np.rot90(restored, axes=(1,2), k=random.randint(1,3)).reshape((1, restored.shape[0], restored.shape[1],restored.shape[2])).astype(np.float16)
        elif r == 4 or r == 5 or r == 6:
            return np.fliplr(restored).reshape((1, restored.shape[0], restored.shape[1],restored.shape[2])).astype(np.float16)
    return restored.reshape((1, restored.shape[0], restored.shape[1],restored.shape[2])).astype(np.float16)

def GetCropPos(aug = False):
    if aug:
        return random.randint(5,10)*10, random.randint(2,8)*10, random.randint(2,8)*10
    else:
        return 70, 70, 70

def GetVolume(pid, aug, path, CTMEAN, CTSTD):
    srcFile = path + 'DCM/Patient-'+str(pid)+'.dcm'
    dataset = pydicom.dcmread(srcFile)
    dataset.file_meta.TransferSyntaxUID
    dataset.BitsAllocated = 16
    frames = decode_data_sequence(dataset.PixelData)
    
    s0, s1, s2 = GetCropPos(aug)
        
    restored = []    
    for f in frames[s0:s0+DEPTH]:
        buf = io.BytesIO(f)
        img = cv2.imdecode(np.frombuffer(buf.getbuffer(), np.ushort), -1)
        restored.append(img)        
    restored = np.array(restored)
    
    restored[restored>2000] = 2000
    restored = restored[:, s1:s1+WIDTH, s2:s2+HEIGHT]
    
    return AugmentNorm(restored, aug, CTMEAN, CTSTD).copy()

def cosine_lr(opt, base_lr, e, epochs):
    lr = 0.5 * base_lr * (math.cos(math.pi * e / epochs) + 1)
    for param_group in opt.param_groups:
        param_group["lr"] = lr
    return max(lr, 0.00005)

def SaveMetrics(y_test, pred, file):
    precision_recall_fscore = []

    prf = precision_recall_fscore_support(y_test, pred,average = "weighted")
    ac = accuracy_score(y_test, pred)

    precision_recall_fscore.append([prf[0],prf[1],prf[2],ac])

    metrics = pd.DataFrame(np.array(precision_recall_fscore), columns=['precision','recall','f1-score','accuracy'])
    mean_values = []
    for i in range(4):
        mean_values.append(np.mean(np.array(precision_recall_fscore)[:,i]))
    metrics = metrics.append(pd.Series(mean_values, index=metrics.columns, name="Average"))
    metrics.to_excel(file)
    return metrics


def accuracy(y, t):
    pred = y.data.max(1, keepdim=True)[1]
    acc = pred.eq(t.data.view_as(pred)).cpu().sum()
    return acc

def to_categorical(y, num_classes=None, dtype='float32'):
    y = np.array(y, dtype='int')
    input_shape = y.shape
    if input_shape and input_shape[-1] == 1 and len(input_shape) > 1:
        input_shape = tuple(input_shape[:-1])
    y = y.ravel()
    if not num_classes:
        num_classes = np.max(y) + 1
    n = y.shape[0]
    categorical = np.zeros((n, num_classes), dtype=dtype)
    categorical[np.arange(n), y] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

class Logger:

    def __init__(self, log_dir, log_file, headers):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        self.f = open(os.path.join(log_dir, log_file), "w")
        header_str = "\t".join(headers + ["EndTime."])
        self.print_str = "\t".join(["{}"] + ["{:.6f}"] * (len(headers) - 1) + ["{}"])

        self.f.write(header_str + "\n")
        self.f.flush()
        print(header_str)

    def write(self, *args):
        now_time = datetime.now().strftime("%m/%d %H:%M:%S")
        self.f.write(self.print_str.format(*args, now_time) + "\n")
        self.f.flush()
        print(self.print_str.format(*args, now_time))

    # def write_hp(self, hp):
    #     json.dump(hp, open(os.path.join(self.log_dir, "hp.json"), "w"))

    def close(self):
        self.f.close()

def roc_plot(n_classes_, y_tests_, y_prediction_proba_, classesnames, colors, path, filename ):
    # plt.rcParams['font.sans-serif']=['Arial']
    plt.figure() 
    plt.rcParams['axes.unicode_minus']=False 
    # plt.grid(linestyle = "--")
    ax = plt.gca()
    # ax.spines['top'].set_visible(False)
    # ax.spines['right'].set_visible(False)
    fig = plt.gcf()
    fig.set_size_inches(5.5, 4.5)

    tprs_all = []
    aucs_all = []
    mean_fpr = np.linspace(0, 1, 100)

    auc_values = []
    for j in range(n_classes_):
        tprs = []
        aucs = []        
        for i in range(len(y_tests_)):
            fpr, tpr, thresholds = roc_curve(to_categorical(y_tests_[i],num_classes=n_classes_)[:, j], y_prediction_proba_[i][:, j])
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs_all.append(np.interp(mean_fpr, fpr, tpr))

            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            aucs_all.append(roc_auc)


        if len(y_tests_)== 1:
            mean_tpr = tprs[0]
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            plt.plot(mean_fpr, mean_tpr, color=colors[j],
                     label=r'%d. %s - ROC (AUC=%0.3f)' % (j+1,classesnames[j], mean_auc),
                     lw=.5, alpha=1)
            auc_values.append(mean_auc)
            
        else:
            mean_tpr = np.mean(tprs, axis=0)
            mean_tpr[-1] = 1.0
            mean_auc = auc(mean_fpr, mean_tpr)
            std_auc = np.std(aucs)
            plt.plot(mean_fpr, mean_tpr, color=colors[j],
                     label=r'%d. %s - Mean ROC (AUC=%0.3f$\pm$%0.3f)' % (j+1, classesnames[j], mean_auc, std_auc),
                     lw=.5, alpha=1)
            auc_values.append(mean_auc)

    mean_tpr = np.mean(tprs_all, axis=0)
    mean_tpr[0] = .0
    mean_tpr[-1] = 1.0
    
    if n_classes_ > 1:
        plt.plot(mean_fpr, mean_tpr, color=colors[-1],
                 label=r'Mean ROC (AUC=%0.3f$\pm$%0.3f)' % (np.mean(auc_values),  np.std(auc_values)),
                 lw=1, alpha=1)
        
    plt.plot([0, 1], [0, 1], linestyle='--', lw=1, color='r',label='Chance', alpha=1)



    std_tpr = np.std(tprs_all, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.1,
                     label=r'$\pm$ 1 std. dev.')

    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])

    plt.xlabel('False Positive Rate',fontsize=10,fontweight='bold')
    plt.ylabel('True Positive Rate',fontsize=10,fontweight='bold')
    plt.legend(loc="lower right")


    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    plt.tight_layout()
    plt.savefig(path + filename + '.svg',format='svg')

    plt.show()
    return mean_tpr, auc_values