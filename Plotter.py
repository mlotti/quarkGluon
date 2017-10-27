#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False
#/////////////////////

import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import keras.backend as K
import pylab as P
import pandas as pd
import numpy as np
import keras.callbacks
import glob

from sklearn.metrics import roc_auc_score,roc_curve,auc
from root_pandas import read_root

from keras.models import Model,load_model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation

listOfFiles=glob.glob("/work/hajohajo/UnsupervisedJets/preprocessed_roots_OpenData/*.root")
#read=['QG_ptD','QG_axis2','QG_mult','jet_eta','isPhysG','isPhysUD']

df = read_root(listOfFiles[-1]) #,columns=read)
df['target'] = (df['isPhysG']==1)
df=df.drop(['isPhysG','isPhysUD'],axis=1)

test_y=df['target']
#df['ratio']=1.0*df[df['target']==1].shape[0]/df.shape[0]
test_x=df.drop('target',axis=1)

to_plot=test_x.columns.values

test_x=test_x.as_matrix()
test_y=test_y.as_matrix()

model=load_model('KERAS_model.h5')

pred_y=model.predict(test_x)

model_sup = load_model('KERAS_model_supervised.h5')

pred_y_sup=model_sup.predict(test_x)

print ' - roc auc: ',round(roc_auc_score(test_y,pred_y),3)

print np.where(test_y==0)

#Bulk plots//////////////////////
"""
from matplotlib.backends.backend_pdf import PdfPages

dir = "./plots/"
dict_ = {'chf':'Charged hadron energy fraction','nhf':'Neutral hadron energy fraction',
	'phf':'Photon energy fraction','elf':'Electron energy fraction',
	'muf':'Muon energy fraction','chm':'Charged hadron multiplicity',
	'nhm':'Neutral hadron multiplicity',
	'phm':'Photon multiplicity','elm':'Electron multiplicity',
	'mum':'Muon multiplicity','beta':'Frac. of chg. had. from hard proc.',
	'jet_area':'Jet area in $\eta-\phi$ -plane'}

bin_dict_ = {'chf':np.arange(0.0,1.0,0.05),'nhf':np.arange(0.0,1.0,0.05),'phf':np.arange(0.0,1.0,0.05),'elf':np.arange(0.0,1.0,0.05),
		'muf':np.arange(0.0,1.0,0.05),'chm':np.arange(0.0,51.0,1.0),'nhm':np.arange(0.0,31.0,1.0),'phm':np.arange(0.0,31.0,1.0),
		'elm':np.arange(0.0,31.0,1.0),'mum':np.arange(0.0,31.0,1.0),'beta':np.arange(0.0,1.0,0.05),'jet_area':np.arange(0.0,1.0,0.025)}

gluons=df[test_y==1]
quarks=df[test_y==0]
fig, axes = plt.subplots(4,3,figsize=(15,20))
index_i = 0
index_j = 0
for column in to_plot:
	histname="Comparison_"+column+".pdf"
	binning = bin_dict_[column]
	print index_i," ",index_j

	axes[index_i,index_j].hist(gluons[column],bins=binning,alpha=0.8,label='Gluons',normed=1)
	axes[index_i,index_j].hist(quarks[column],bins=binning,alpha=0.8,label='Quarks',normed=1)


        axes[index_i,index_j].legend()
        axes[index_i,index_j].set_title("Comparison of "+dict_[column])
        axes[index_i,index_j].set_ylabel('Jet fraction')
        axes[index_i,index_j].set_xlabel(dict_[column])
        index_j=index_j+1
	if (index_j%3 == 0 and index_j!=0):
		index_i=index_i+1
		index_j=0

fig.tight_layout()
plt.savefig('Distributions.png')
"""
#///////////////////////////////

plt.clf()
gluons = pred_y[np.where(test_y==1)]
quarks = pred_y[np.where(test_y==0)]
#binning = np.linspace(0,1.0,20)
binning = np.arange(0.0,1.0,0.05)
print binning
plt.hist(gluons,bins=binning,alpha=0.8,label='Gluons',normed=1)
plt.hist(quarks,bins=binning,alpha=0.8,label='Quarks',normed=1)
plt.legend()
plt.title('Quark-Gluon classifier, weakly supervised')
plt.xlabel('MVA output')
plt.ylabel('Jets')
plt.savefig('weakClassif_distr.png')
plt.clf()

gluons = pred_y_sup[np.where(test_y==1)]
quarks = pred_y_sup[np.where(test_y==0)]
binning = np.arange(0.0,1.0,0.05)
plt.hist(gluons,bins=binning,alpha=0.8,label='Gluons',normed=1)
plt.hist(quarks,bins=binning,alpha=0.8,label='Quarks',normed=1)
plt.legend()
plt.title('Quark-Gluon classifier, supervised')
plt.xlabel('MVA output')
plt.ylabel('Jets')
plt.savefig('supClassif_distr.png')

#ROC curve for plotting
fpr,tpr, thresholds  = roc_curve(test_y,pred_y)
roc_auc = auc(fpr, tpr)

fpr_sup,tpr_sup,thresholds_sup = roc_curve(test_y,pred_y_sup)
roc_auc_sup = auc(fpr_sup,tpr_sup)

plt.clf()
plt.plot(fpr,tpr,'b',label='Weakly sup. AUC = %0.2f'% roc_auc)
plt.plot(fpr_sup,tpr_sup,'r--',label='Sup. AUC = %0.2f'% roc_auc_sup)
plt.plot([0,1],[0,1],'k--')
plt.xlim([0.0,1.0])
plt.ylim([0.0,1.0])
plt.legend(loc='lower right')
plt.title("Receiver Operating Characteristic")
plt.ylabel('True positive rate')
plt.xlabel('False positive rate')
plt.savefig('roc_curve.png')
