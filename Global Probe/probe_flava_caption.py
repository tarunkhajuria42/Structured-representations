from sklearn import *
import os
import pickle
from sklearn.utils import shuffle
from sklearn.linear_model import Perceptron
from probe_utils import *
import sys
from pycocotools.coco import COCO


train_dir = 'Features/FLAVA/Train/'
val_dir ='Features/FLAVA/Val/'
output_file = 'Global_Probe_caption_FLAVA.pickle'

token_types = ['random_features','all_features']
token_labels = ['random_obj','avg_obj']

rand_set = [20, 23, 52, 64, 51, 70, 19, 53, 11, 21, 32, 58, 14, 31, 36, 56, 25, 13,  8, 46]

# assemble the data and the train the decoder
import numpy as np
# lets do it layer wise
results = {}
layers = np.arange(12)
for l in layers:
    print(l)
    results[l] = {}
    train_holder = get_features_for_layer(train_dir, l,rand_set)
    val_holder = get_features_for_layer(val_dir, l,rand_set,val=True)
    print('data_done')
    y_train = train_holder['label']
    y_val = val_holder['label']
    y_val_nc = val_holder['nc_label']
    y_test = []
    for token in token_types:
        X_train = train_holder[token]
        X_val = val_holder[token]
        X_val_nc = val_holder['nc_'+token]
        X_test = []
        res,res_nc = model_results(X_train,y_train,X_val,y_val,X_val_nc,y_val_nc,caption=True)
        print(token,res,res_nc)
        results[l][token]= res
        results[l]['nc_'+token]= res_nc
with open('Results/'+output_file,'wb') as f:
    pickle.dump(results,f)

