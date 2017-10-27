import root_pandas
import pandas as pd
import os
import shutil
import numpy as np

"""
Parameters for the preprocessing script
path_ - the folder where the root files and the text file containing the sample names are
fname_ - text file with one input root file name per row
read - SCALAR variables to be read from the root file (i.e. variables with one entry per event)
flattens - VECTOR variables to be read from the root file (i.e. variables with multiple entries per event)
outFolder_ - path to output folder where the files will be saved. Careful, overwrites existing folder.
"""

path_ = "/work/data/QCD_reordered_50_80GeV/"
fname = path_+"train_val_samples.txt"
read=['QG_ptD','QG_axis2','QG_mult','Cpfcan_pt','jet_eta','isPhysG','isPhysUD']
flattens_=['Cpfcan_pt','Cpfcan_eta','Cpfcan_phi','Npfcan_pt','Npfcan_eta','Npfcan_phi']
outFolder_="/work/hajohajo/UnsupervisedJets/preprocessed_roots"

if os.path.exists(outFolder_):
	shutil.rmtree(outFolder_)
os.makedirs(outFolder_)
read=read+flattens_
with open(fname) as f:
	content = f.readlines()
content = [x.strip() for x in content] 

#Contains list of filepaths
content = [path_ + s for s in content]
content = ['OpenDataTree_mc.root']

#Function to flatten columns that contain lists as entries (for example Cpfcan_pt)
def flattencolumns(df1, cols,len):
	df = pd.concat([pd.DataFrame(df1[x].values.tolist()).add_prefix(x).iloc[:,:len] for x in cols], axis=1)
	df.fillna(0.0,inplace=True)
	df1.drop(cols, axis=1,inplace=True)
	df.reset_index(drop=True, inplace=True)
	df1.reset_index(drop=True, inplace=True)
	return pd.concat([df, df1], axis=1)

read=['QG_ptD','QG_axis2','QG_mult','jet_eta','isPhysG','isPhysUD']
flattens_=['Cpfcan_pt','Cpfcan_eta','Cpfcan_phi','Npfcan_pt','Npfcan_eta','Npfcan_phi']
read=read+flattens_

#Read in the files one at a time, perform preprocessing and save the new file as to a folder for
#the analysis

counter=1
#saveName=outFolder_+"/preprocessed_files.root"
#if os.path.exists(saveName):
#	os.remove(saveName)
for file in content:
	df = root_pandas.read_root(file,'deepntuplizer/tree',columns=read)
	#Skimming conditions! Check that these match what you want to include in your analysis
	num_pfCands=10
	df = df[(df.isPhysG == 1)|(df.isPhysUD == 1)]
	df = df[(np.abs(df.jet_eta) < 1.3)]
	df = flattencolumns(df,flattens_,num_pfCands)

	saveName=outFolder_+"/preprocessed_"+file.rsplit('/',1)[-1]
	df.to_root(saveName,key='tree')
	print("Processed: "+str(counter)+"/"+str(len(content)))
	counter=counter+1

