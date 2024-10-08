import sklearn
import os
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_digits
from sklearn.linear_model import Perceptron
from sklearn.metrics import f1_score
from PIL import Image
from matplotlib import pyplot as plt
from nltk.stem import WordNetLemmatizer

np.random.seed(0)

def balance(Ys):
    feature_type_norms = ['random_features','all_features','all_random','cls_features']
    cats_1 = np.unique(np.array(Ys[feature_type_norms[0]][1]))
    cats_2 = np.unique(np.array(Ys[feature_type_norms[0]][2]))
    categories = np.array(Ys[feature_type_norms[0]][3])
    unique_cats = np.unique(categories)
    features_cat= {}
    inds = []
    for cat in cats_2:
        obj1 = np.where(cats_1[0]+'_'+cat == categories)[0]
        obj2 = np.where(cats_1[1]+'_'+cat == categories)[0]
        if(len(obj1)>len(obj2)):
            obj1 = np.random.choice(obj1,replace = False,size = len(obj2))
        else:
            obj2 = np.random.choice(obj2,replace = False,size = len(obj1))
        inds.extend(obj1)
        inds.extend(obj2)
    inds = np.array(inds)
    return inds

def data_preprocess(x,y):
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True,stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True,stratify=y_train)
    return X_train,y_train,X_val,y_val,X_test,y_test
    
def model_results(X_train,y_train,X_val,y_val,X_test,y_test,final=False,f1 = False):
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X_train, y_train)
    if(final):
        if(f1):
            preds = clf.predict(X_test)
            return f1_score(y_test,preds)
        else:    
            return clf.score(X_test, y_test)
    else:
        if(f1):
            preds = clf.predict(X_val)
            return f1_score(y_val,preds)
        else:
            return clf.score(X_val, y_val)
            
layers = np.arange(0,12)
feature_type = ['random_features','all_features','all_random','cls_features']

input_dir = 'Features_VIT/'
files = os.listdir(input_dir)
out_dir = 'Results_Balanced_VIT/'
if(not os.path.isdir(out_dir)):                        
    os.mkdir(out_dir)
for task in range(6):
    results ={}
    cat = {}
    for l in layers:
        print(task,l)
        Xs = {}
        Ys = {}
        results[l]={}
        for f in feature_type:
            Xs[f]={}
            Ys[f]={}
            for ind,file in enumerate(files):
                file_parts = file.split('_')
                if(len(file_parts)<4):
                    continue
                t_no = int(file_parts[1])
                if(t_no != task):
                    continue
                with open(input_dir+file,'rb') as fid:
                    object = pickle.load(fid)
                cat[1] = file_parts[2]
                cat[2] = file_parts[3]
                cat[3] = cat[1] + '_' + cat[2]
                for obj in object:
                    for x in [1,2]:
                        x_sample = obj[cat[x]][f][l]
                        if( x in Xs[f].keys()):
                            Xs[f][x].append(x_sample)
                        else:
                            Xs[f][x] =[]
                            Xs[f][x].append(x_sample)
                            
                    for y in [1,2,3]:
                        if(y in Ys[f].keys()):
                            Ys[f][y].append(cat[y])
                        else:
                            Ys[f][y] =[]
                            Ys[f][y].append(cat[y])
        print('data done')
        inds = balance(Ys)
        for f in feature_type:
                results[l][f]={}
                for x in [1,2]:
                    results[l][f][x]= {}
                    for y in [1,2,3]:
                        x_data,y_data = np.array(Xs[f][x]),np.array(Ys[f][y])
                        x_data,y_data = x_data[inds],y_data[inds]
                        X_train,y_train,X_val,y_val,X_test,y_test = data_preprocess(x_data,y_data)
                        results[l][f][x][y]= model_results(X_train,y_train,X_val,y_val,X_test,y_test,final=True)
    with open(out_dir +'task{}_results.pkl'.format(task),'wb') as f:
        pickle.dump(results,f)