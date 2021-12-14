# %%
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  accuracy_score
from joblib import dump, load
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from .utils import *

# %%
path = '/guoqing/project/covid/'

# %%
basicInfo = pd.read_pickle(path + 'info.ppl')

#class 0 : Control, Suspected ; class 1: Mild, Regular; class 2: Severe, Critically ill
def GetMorbidityClass(morbidity):
    if morbidity == 'Control' or morbidity == 'Suspected':
        return 0
    elif morbidity == 'Mild' or morbidity == 'Regular':
        return 1
    elif morbidity == 'Severe' or morbidity == 'Critically ill':
        return 2
    else:
        assert(False)
        
morbidityClasses = [GetMorbidityClass(morbidity) for morbidity in basicInfo.Morbidity]


# %%
basicInfo.loc[:, 'COVIDSeverity'] = morbidityClasses

# %%
radiofeatures = pd.read_pickle(path + 'radioinfo.ppl')

# %%
radiofeatures.loc[:, 'COVIDNC'] = list(basicInfo.COVIDNC)
radiofeatures.loc[:, 'COVIDCT'] = list(basicInfo.COVIDCT)
radiofeatures.loc[:, 'COVIDSeverity'] = list(basicInfo.COVIDSeverity)

# %%
# radiofeatures = radiofeatures[radiofeatures.PID.isin(list(basicInfo[basicInfo.Morbidity!='Suspected'].PID))]

# %%

scaler = MinMaxScaler()
scaler.fit(radiofeatures.iloc[:,1:-3])
scaledfeatures = pd.DataFrame(scaler.transform(radiofeatures.iloc[:,1:-3]),columns = radiofeatures.columns[1:-3])

# %%
scaledfeatures.loc[:, 'PID'] = list(radiofeatures.PID)
scaledfeatures.loc[:, 'COVIDNC'] = list(radiofeatures.COVIDNC)
scaledfeatures.loc[:, 'COVIDCT'] = list(radiofeatures.COVIDCT)
scaledfeatures.loc[:, 'COVIDSeverity'] = list(radiofeatures.COVIDSeverity)


# %% [markdown]
# # Performance using LGBM

x_train, x_test, y_train_nc, y_test_nc = train_test_split(scaledfeatures.iloc[:,:-4], scaledfeatures.COVIDNC,stratify=scaledfeatures.COVIDNC, test_size=0.30, random_state=11, shuffle=True)

y_train_ct = scaledfeatures.loc[y_train_nc.index].COVIDCT
y_test_ct = scaledfeatures.loc[y_test_nc.index].COVIDCT

y_train_severity = scaledfeatures.loc[y_train_nc.index].COVIDSeverity
y_test_severity = scaledfeatures.loc[y_test_nc.index].COVIDSeverity

subpath = path + '.results/lgbm/'

classesnames = ['Negative', 'Positive']
colors = ['blue', 'yellow', 'green']
models = []
for targets, name in zip([(y_train_ct, y_test_ct), (y_train_nc, y_test_nc), (y_train_severity, y_test_severity)], ['ct', 'nc', 'sv']):
    print(name)
#     x_train, x_test, y_train, y_test = train_test_split(scaledfeatures.iloc[:,1:-4], target,stratify=scaledfeatures.COVIDNC, test_size=0.30, random_state=11, shuffle=True)
    
    model = LGBMClassifier(boosting_type='gbdt', objective='multiclass' if name == 'sv' else 'binary',
                       num_class= 3 if name == 'sv' else 1,
                       n_estimators=1000 , learning_rate=0.01)

    model.fit(x_train, targets[0])
    models.append(model)
    dump(model, subpath + 'lgbm_'+name+'.model')
    proba = model.predict_proba(x_test)

    ac = accuracy_score(targets[1], np.argmax(proba,axis=1))
    print("Test Accuracy Againt COVID ",name, ac)
    if name =='sv':
        classesnames = ['Control/Suspected', 'Mild/Regular', 'Severe/Critically ill']
        colors = ['blue', 'yellow', 'red', 'green']

    # the test roc/auc
    mean_tpr, auc_values = roc_plot(3 if name == 'sv' else 2,[targets[1]],[proba], classesnames, colors, subpath, 'roc_lgbm_'+name)
    np.save(subpath + 'roc_lgbm_'+name+'_mean_tpr.npy',mean_tpr)
    np.save(subpath + 'roc_lgbm_'+name+'_auc_values.npy',auc_values)
    SaveMetrics(targets[1], np.argmax(proba,axis=1), subpath + 'metrics_lgbm_'+name+'.xlsx')

# %% [markdown]
# # Select top radiomic features

# %%
pds = pd.DataFrame(x_train.columns, columns=['features'])

pds.loc[:,'ct_importance'] = models[0].feature_importances_
pds.loc[:,'nc_importance'] = models[1].feature_importances_
pds.loc[:,'sv_importance'] = models[2].feature_importances_

# %%
subpath = path + '.results/lgbm/'
pds.to_excel(subpath + 'radiofeature_importance.xlsx')

# %%
ctf = list(pds.sort_values(by='ct_importance', ascending=False)[:10].features)
ncf = list(pds.sort_values(by='nc_importance', ascending=False)[:10].features)
svf = list(pds.sort_values(by='sv_importance', ascending=False)[:10].features)

# %%
lst=[]
lst.extend(ctf)
lst.extend(ncf)
lst.extend(svf)

topfeatures = list(set(lst))

# %%
selfeatures = ['PID']
selfeatures.extend(topfeatures)
selfeatures.extend(['COVIDNC','COVIDCT','COVIDSeverity'])
topfeatureframe = radiofeatures.loc[:, selfeatures]
topfeatureframe.loc[:, 'Age'] = list(basicInfo.Age)
topfeatureframe.loc[:, 'Gender'] = list(basicInfo.Gender)

topfeatureframe.to_excel(subpath + 'topfeatures.xlsx')

tenSignificantFeatures = [
 'Maximum',
 'LLL_glcm_Imc1',
    'LHH_glcm_ClusterShade',
 'HLH_glcm_ClusterShade',
 'GrayLevelVariance',
 'LLL_glcm_Imc2',
    'HLL_glcm_DifferenceVariance',
 'Imc2',
 'LLL_glcm_Correlation',
 'Imc1']

# %% [markdown]
# # Boxplot top features

# %%

subpath = path + 'data/'
pds = pd.read_excel(subpath + 'topfeatures.xlsx')
topfeatures = list(pds.columns[2:-5])
sel = [1,2,4,9,15,17,18,19,21,22]
for fc in ['COVIDCT', 'COVIDNC', 'COVIDSeverity']:
    fcs = tenSignificantFeatures.copy()
    fcs.extend([fc])
    dfbox = scaledfeatures.loc[:, fcs]
    a = list(dfbox.columns)
    a[6] = '*DifferenceVariance'
    dfbox.columns = a
    df_long = pd.melt(dfbox, fc, var_name="Features", value_name="Scaled Values")
    fig, ax = plt.subplots(figsize=(20,  5))
    sns.boxplot(x="Features", hue=fc, y="Scaled Values", data=df_long, ax=ax, palette="Set2", linewidth=2.5, fliersize=2, saturation=1)
    ax.xaxis.set_tick_params(labelsize=10)
    ax.yaxis.set_tick_params(labelsize=10)
    # plt.xticks(rotation=90)
    plt.xlabel('Features',fontsize=14)
    plt.ylabel('Scaled Values',fontsize=14)
    L=plt.legend(loc='lower right', prop={'size': 14})
    
    if 'COVIDSeverity' == fc:
        L.get_texts()[0].set_text('Control/Suspected')
        L.get_texts()[1].set_text('Mild/Regular')
        L.get_texts()[2].set_text('Severe/Critically ill')
    else:
        L.get_texts()[0].set_text('Negative')
        L.get_texts()[1].set_text('Positive')
    plt.tight_layout()
    plt.savefig(path + '.results/boxplot_'+fc+'.svg',format='svg')
    plt.show()

# %% [markdown]
# # Case study

# %%
grayscale = True

# %%
subpath = path + 'data/'
topfeatureframe = pd.read_excel(subpath + 'topfeatures.xlsx')
twoframeRaw = topfeatureframe[topfeatureframe.PID.isin([17,567])][tenSignificantFeatures]
a = list(twoframeRaw.columns)
a[6] = '*DifferenceVariance'
twoframeRaw.columns = a
twoframe = twoframeRaw.copy()
twoframe /= twoframe.max()
twoframe /= twoframe.max()
twoframe.index = ['Infected','Normal']
fig, ax = plt.subplots(figsize=(4,  6))
if grayscale:
    ax = twoframe.T.plot(kind='barh', ax=ax, color=('k','0.5'))
else:
    ax = twoframe.T.plot(kind='barh', ax=ax, color=(sns.color_palette("Set2")[1],sns.color_palette("Set2")[0]))

ax.set_xlim(0.20, 1.01)
ax.set_xlabel('Scaled Feature Values')
ax.xaxis.set_tick_params(labelsize=10)
ax.yaxis.set_tick_params(labelsize=10)
L=plt.legend(loc='lower right', prop={'size': 10})
# plt.tight_layout()
if grayscale:
    plt.savefig(path + '.results/barplot_casecompare_gray.svg',format='svg')
else:
    plt.savefig(path + '.results/barplot_casecompare.svg',format='svg')

plt.show()


# # Performance using RF
classesnames = ['Negative', 'Positive']
colors = ['blue', 'yellow', 'green']
subpath = path + '.results/rf/'

for targets, name in zip([(y_train_ct, y_test_ct), (y_train_nc, y_test_nc), (y_train_severity, y_test_severity)], ['ct', 'nc', 'sv']):
    print(name)
#     x_train, x_test, y_train, y_test = train_test_split(scaledfeatures.iloc[:,1:-4], target,stratify=scaledfeatures.COVIDNC, test_size=0.30, random_state=11, shuffle=True)

    model = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=11)  
    model = model.fit(x_train, targets[0])
    dump(model, subpath + 'rf_'+name+'.model')

    proba = model.predict_proba(x_test)

    ac = accuracy_score(targets[1], np.argmax(proba,axis=1))
    print("Test Accuracy Againt COVID ",name, ac)
    if name =='sv':
        classesnames = ['Control/Suspected', 'Mild/Regular', 'Severe/Critically ill']
        colors = ['blue', 'yellow', 'red', 'green']
    # the test roc/auc
    mean_tpr, auc_values = roc_plot(3 if name == 'sv' else 2,[targets[1]],[proba], classesnames, colors, subpath, 'roc_rf_'+name)
    np.save(subpath + 'roc_rf_'+name+'_mean_tpr.npy',mean_tpr)
    np.save(subpath + 'roc_rf_'+name+'_auc_values.npy',auc_values)
    SaveMetrics(targets[1], np.argmax(proba,axis=1), subpath + 'metrics_rf_'+name+'.xlsx')

# %%



