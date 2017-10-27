#Restrict to one gpu
import imp
try:
	imp.find_module('setGPU')
	import setGPU
except ImportError:
	found = False

#Import modules
import tensorflow as tf
sess = tf.Session()
import matplotlib.pyplot as plt
import keras.backend as K
K.set_session(sess)
import keras.callbacks
import pylab as P
import numpy as np
import pandas as pd
import root_pandas
import glob
import math
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

loss_ = 'binary_crossentropy'

#/////////////////TO BE MOVED INTO SEPARATE FILE FOR CLARITY
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
#	string = str(round(logs.get('loss'),4))+"\t"+str(round(logs.get('val_loss'),4))+"\t"+str(round(logs.get('acc'),4))+"\t"+str(round(logs.get('val_acc'),4))+"\t"+str(round(roc_auc_score(test_y,self.model.predict(test_x)),3))+"\n"
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

# listOfFiles=glob.glob("/work/data/preprocessed_roots_OpenData/*.root")
listOfFiles=["/work/data/preprocessed_roots_OpenData/preprocessed_OpenDataTree_mc_final150000.root", "/work/data/preprocessed_roots_OpenData/preprocessed_OpenDataTree_mc_final165000.root"]
df = root_pandas.read_root(listOfFiles, 'tree')

# Split the data to train and test sets
train_x, test_x = train_test_split(df, test_size=0.3)

# Extract the labels
train_y = train_x.isPhysG.copy()
test_y = test_x.isPhysG.copy()

train_x.drop(['isPhysG','isPhysUD'], axis=1, inplace=True)
test_x.drop(['isPhysG','isPhysUD'], axis=1, inplace=True)

# Convert the dataframes to matrices
train_x = train_x.as_matrix()
train_y = train_y.as_matrix()
test_x = test_x.as_matrix()
test_y = test_y.as_matrix()

# Build the neural network
from keras.models import Sequential,Model
from keras.layers import Input,Dense,Activation,Dropout
from keras.constraints import maxnorm
from keras import optimizers
from sklearn.utils import class_weight

model = Sequential()
model.add(Dense(200, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3), input_dim=train_x.shape[1]))
model.add(Dropout(0.1))
model.add(Dense(200, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.1))
model.add(Dense(50, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.1))
model.add(Dense(30, kernel_initializer='normal', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

model.compile(optimizer='Nadam', loss=loss_, metrics=['accuracy'])

cb = ROC_value()
loss = LossHistory()
classWeight = class_weight.compute_class_weight('balanced', np.unique(train_y), train_y[:])
batchSize = 1000
numberOfEpochs = 100
model.fit(train_x, train_y,
        epochs=numberOfEpochs,
        batch_size = batchSize,
        class_weight=classWeight,
        callbacks=[cb,loss],
        validation_split=0.1,
        shuffle=True)


model.save('supervised_model_'+loss_+'_Nadam.h5')
