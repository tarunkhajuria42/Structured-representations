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

def data_preprocess(x,y):
    ''' Pre-process data into to train/val/test splits'''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42, shuffle=True,stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42, shuffle=True,stratify=y_train)
    return X_train,y_train,X_val,y_val,X_test,y_test
    
def model_results(X_train,y_train,X_val,y_val,X_test,y_test,final=False,f1 = False):
    ''' function to get the results for a particular set'''
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

wnl = WordNetLemmatizer()
def word_in_caption(word,caption):
    caption = caption.lower().split()
    word = wnl.lemmatize(word.lower())
    caption = list(map(wnl.lemmatize, caption))
    return (word in caption)
            
layers = np.arange(0,12)
feature_type = ['random_features','all_features','all_random','cls_features']

input_dir = 'Features_BLIP/'
files = os.listdir(input_dir)

if( not os.path.isdir('Results_BLIP/')):
    os.mkdir('Results_BLIP/')

for task in range(6):
    results ={}
    cat = {}
    for l in layers:
        print(l)
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
                        caption = obj[cat[x]]['caption']
                        in_caption = word_in_caption(cat[x],caption)
                        if( x in Xs[f].keys()):
                            Xs[f][x]['vector'].append(x_sample)
                            Xs[f][x]['caption'].append(in_caption)
                        else:
                            Xs[f][x] = {}
                            Xs[f][x]['vector'] =[]
                            Xs[f][x]['caption'] =[]
                            Xs[f][x]['vector'].append(x_sample)
                            Xs[f][x]['caption'].append(in_caption)
                            
                    for y in [1,2,3]:
                        if(y in Ys[f].keys()):
                            Ys[f][y].append(cat[y])
                        else:
                            Ys[f][y] =[]
                            Ys[f][y].append(cat[y])
        print('data done')
        for f in feature_type:
                results[l][f]={}
                for x in [1,2]:
                    results[l][f][x] = {}
                    results[l][f][x][True]= {}
                    results[l][f][x][False]= {}
                    x_data,caption = np.array(Xs[f][x]['vector']),Xs[f][x]['caption']
                    index_caption = np.where(caption)[0]
                    index_no_caption = np.where(np.invert(caption))[0]
                    x_data_caption = x_data[index_caption]
                    x_data_no_caption = x_data[index_no_caption]
                    for y in [1,2,3]:
                        y_data = np.array(Ys[f][y])[index_caption]
                        if(len(y_data)<400):
                            results[l][f][x][True][y] = None
                            results[l][f][x][False][y] = None
                            continue
                        try:
                            X_train,y_train,X_val,y_val,X_test,y_test = data_preprocess(x_data_caption,y_data)
                            results[l][f][x][True][y]= model_results(X_train,y_train,X_val,y_val,X_test,y_test,final=True,f1=False)
                        except:
                            results[l][f][x][True][y] = None
                            results[l][f][x][False][y] = None
                            continue
                        y_data = np.array(Ys[f][y])[index_no_caption]
                        if(len(y_data)<400):
                            results[l][f][x][True][y] = None
                            results[l][f][x][False][y] = None
                            continue
                        try:
                            X_train,y_train,X_val,y_val,X_test,y_test = data_preprocess(x_data_no_caption,y_data)
                            results[l][f][x][False][y]= model_results(X_train,y_train,X_val,y_val,X_test,y_test,final=True,f1=False)
                        except:
                            results[l][f][x][True][y] = None
                            results[l][f][x][False][y] = None
                            continue
    with open('Results_BLIP/task{}_caption.pkl'.format(task),'wb') as f:
        pickle.dump(results,f)

