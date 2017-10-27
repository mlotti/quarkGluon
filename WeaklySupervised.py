#Restrict to one gpu
import imp
try:
        imp.find_module('setGPU')
        import setGPU
except ImportError:
        found = False
#/////////////////////
import tensorflow as tf
sess = tf.Session()
import matplotlib.pyplot as plt
import keras.backend as K
K.set_session(sess)
import pylab as P
import pandas as pd
import root_pandas
import ROOT
import numpy as np
import keras.callbacks
import glob
import math
from sklearn.metrics import roc_auc_score

loss_ = 'binary_crossentropy'

#/////////////////TO BE MOVED INTO SEPARATE FILE FOR CLARITY
#Custom loss for weakly supervised learning
def weakSquaredLoss(y_true,y_pred):
	return K.mean(K.abs(y_pred - y_true))


#ROC value to be printed out after epochs. Does not affect training
class ROC_value(keras.callbacks.Callback):
        def on_epoch_end(self, batch,logs={}):
                print ' - roc auc: ',round(roc_auc_score(test_y,self.model.predict(test_x)),3)

#Save losses etc. to a separate text file for plotting later
class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
#        file=open("Losses_"+loss_+".txt","w+")
#        file.close()

        with open("Losses_"+loss_+".txt","a") as myfile:
                myfile.write('\n\n Model information:\n')
                self.model.summary(print_fn=lambda x: myfile.write(x + '\n'))
                myfile.write('\n\n Loss Accuracy        Val.loss        Val.Accuracy    ROC AUC\n')


    def on_epoch_end(self, batch, logs={}):

        string = str(round(logs.get('loss'),4))+"\t"+str(round(logs.get('val_loss'),4))+"\n"
        with open("Losses_"+loss_+".txt","a") as myfile:
                myfile.write(string)


def list_columns(obj, cols=4, columnwise=True, gap=4):
    """
    Print the given list in evenly-spaced columns.

    Parameters
    ----------
    obj : list
        The list to be printed.
    cols : int
        The number of columns in which the list should be printed.
    columnwise : bool, default=True
        If True, the items in the list will be printed column-wise.
        If False the items in the list will be printed row-wise.
    gap : int
        The number of spaces that should separate the longest column
        item/s from the next column. This is the effective spacing
        between columns based on the maximum len() of the list items.
    """

    sobj = [str(item) for item in obj]
    if cols > len(sobj): cols = len(sobj)
    max_len = max([len(item) for item in sobj])
    if columnwise: cols = int(math.ceil(float(len(sobj)) / float(cols)))
    plist = [sobj[i: i+cols] for i in range(0, len(sobj), cols)]
    if columnwise:
        if not len(plist[-1]) == cols:
            plist[-1].extend(['']*(len(sobj) - len(plist[-1])))
        plist = zip(*plist)
    printer = '\n'.join([
        ''.join([c.ljust(max_len + gap) for c in p])
        for p in plist])
    return printer



#/////////////////////////////////////////////

#listOfFiles=glob.glob("/work/hajohajo/UnsupervisedJets/preprocessed_roots/*.root")
listOfFiles=glob.glob("/work/hajohajo/UnsupervisedJets/preprocessed_roots_OpenData/*.root")

#read=['QG_ptD','QG_axis2','QG_mult','jet_eta','isPhysG','isPhysUD']
df = root_pandas.read_root(listOfFiles[:-1],'tree') #,columns=read)
df['target'] = (df['isPhysG']==1)
df['target'] = df['target'].apply(lambda row: int(row))

gluons = df[(df.isPhysG == 1)]
quarks = df[(df.isPhysUD == 1)]
gluons=gluons.drop(['isPhysG','isPhysUD'],axis=1)
quarks=quarks.drop(['isPhysG','isPhysUD'],axis=1)

#Need to split the events into at least two bags with different fractions. These
#represent different physical processes where we have a theoretical prediction
#on the ratio of quark and gluon jets

from sklearn.model_selection import train_test_split

gl1,gl2 = train_test_split(gluons,test_size=0.1,random_state=7)
gl2,gl3 = train_test_split(gl2,test_size=0.2,random_state=4)
qu1,qu2 = train_test_split(quarks,test_size=0.9,random_state=42)
qu2,qu3 = train_test_split(qu2,test_size=0.8,random_state=19)

df1=pd.concat([gl1,qu2],ignore_index=True)
df2=pd.concat([gl2,qu1],ignore_index=True)
df3=pd.concat([gl3,qu3],ignore_index=True)

ratio1_ = 1.0*df1[df1['target']==1].shape[0]/df1.shape[0]
ratio2_ = 1.0*df2[df2['target']==1].shape[0]/df2.shape[0]
ratio3_ = 1.0*df3[df3['target']==1].shape[0]/df3.shape[0]

print ratio1_
print ratio2_
print ratio3_

df1['ratio']=ratio1_
df2['ratio']=ratio2_
df3['ratio']=ratio3_

df=pd.concat([df1,df2,df3],ignore_index=True)
#df=pd.concat([df1,df2],ignore_index=True)

train_x, test_x = train_test_split(df,test_size=0.1,random_state=7)

test_y=test_x['target']
train_y=train_x['ratio']
train_x=train_x.drop(['target','ratio'],axis=1)
test_x=test_x.drop(['target','ratio'],axis=1)


train_x=train_x.as_matrix()
test_x=test_x.as_matrix()
train_y=train_y.as_matrix()
test_y=test_y.as_matrix()

#Ratio of samples
ratio_ = 1.0*(train_y==1).sum()/len(train_y)

from keras.models import Model
from keras.layers import Input,Dense,Convolution1D,Flatten,Dropout,Activation
import keras.backend as K
from sklearn.utils import class_weight
from keras.layers.normalization import BatchNormalization

#Create a file to save info of the training
file=open("Losses_"+loss_+".txt","w+")
file.close()

with open("Losses_"+loss_+".txt","a") as myfile:
        myfile.write("\n\n Used variables:\n")
        myfile.write(list_columns(df.columns.values,cols=4))

#Defining the network topology

dropoutRate=0.1
a_inp = Input(shape=(train_x.shape[1],),name='ins')

a = Dense(300,activation='relu', kernel_initializer='normal')(a_inp)
a = Dropout(dropoutRate)(a)
a = Dense(200,activation='relu', kernel_initializer='normal')(a)
a = Dropout(dropoutRate)(a)
a = Dense(30,activation='relu', kernel_initializer='normal')(a)
a = Dropout(dropoutRate)(a)
a = Dense(10,activation='relu', kernel_initializer='normal')(a)
a_out = Dense(1, activation='sigmoid', kernel_initializer='normal',name='outs')(a)

model=Model(inputs=a_inp,outputs=a_out)
"""
b_inp = Input(shape=(train_x.shape[1],),name='ins')
b = Dense(30,activation='relu', kernel_initializer='normal')(b_inp)
b_out = Dense(1, activation='sigmoid', kernel_initializer='normal',name='outs')(b)
model=Model(inputs=b_inp,outputs=b_out)
"""


from keras import optimizers
adam=optimizers.Adam(lr=10.0)
#model.compile(loss=loss_,optimizer=adam,metrics=['acc'])
#loss_ = weakSquaredLoss
batchS=10000
print ratio_
model.compile(loss=loss_,optimizer="Nadam") #,metrics=["acc"]) #['acc'])


cb=ROC_value()
loss=LossHistory()
#check=keras.callbacks.ModelCheckpoint('KERAS_best_model_'+loss_+'.h5',monitor='val_loss',save_best_only=True)
class_weight = class_weight.compute_class_weight('balanced', np.unique(train_y),train_y[:])
#class_weight = class_weight.compute_class_weight('balanced',np.unique(algos),algos[:])
Nepoch=100
model.fit(train_x,train_y,
        epochs=Nepoch,
        batch_size=batchS,
        class_weight=class_weight,
        callbacks=[cb,loss],
#	callbacks=[cb],
        validation_split=0.1,
        shuffle=True)

model.save('my_model_'+loss_+'_Adam.h5')

