import sklearn
import os
import pickle
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.linear_model import Perceptron
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt

layers = np.arange(0,12)
feature_type = ['random_features','all_features','all_random','cls_features']

x_ticks = ['random_obj','avg_obj','random','CLS']
y_labels = ['Token from primary object','Token from second object']
x_labels = ['Decoding primary object','Decoding secondary object','Decoding combination of both']

results = {}
f_legend = ['random_obj','avg_obj','random','CLS/avg']
network_title = ['CLIP-VIT','CLIP-L','BLIP','BLIP-L','FLAVA','CLIP-RenetX4','VIT','DINO','DINO2']
colors = ['b','g','r','c']
results = {}
for network in ['BLIP','BLIP_Large','CLIP','CLIP_Large','FLAVA','CNN','VIT','DINO','DINO2']:
    results[network] = {}
    for task in range(6):
        with open('Results_Balanced_{}/task{}_results.pkl'.format(network,task),'rb') as f:
            results[network][task] = pickle.load(f)
fig, axs = plt.subplots(9, 2,figsize=(13, 15))
fs =[]
for no, network in enumerate(['CLIP','CLIP_Large','BLIP','BLIP_Large','FLAVA','CNN','VIT','DINO','DINO2']):
    features = ['random_features','all_features','all_random','cls_features']
    if(network == 'CNN'):
        features = ['random_features','all_features','all_random','avg']
    for ind, feature in enumerate(features):
        fs.append([])
        layers = results[network][task].keys()
        for obj in [1,2]:
            data = np.zeros((6,len(layers)))
            for layer in results[network][task].keys():
                for task in range(6):
                    if(network == 'CNN'):
                        data[task][layer-1]= results[network][task][layer][feature][obj][obj]    
                    else:
                        data[task][layer]= results[network][task][layer][feature][obj][obj]
            mean = np.mean(data,axis =0)
            sd = np.std(data,axis =0)
            fs[ind].append(axs[no,obj-1].plot(layers,mean,colors[ind],label =f_legend[ind])[0])
            #fs[ind].append(axs[no,obj-1].fill_between(layers,mean-sd,mean+sd,facecolor= colors[ind],label =f_legend[ind],alpha=0.3))
            axs[no,obj-1].fill_between(layers,mean-sd,mean+sd,facecolor= colors[ind],label =f_legend[ind],alpha=0.3)
            axs[no,0].set_ylabel(network_title[no],fontsize = 14)
            axs[no,obj-1].set_ylim([0.2, 1.1])
            axs[6,obj-1].set_xlabel('Object {}'.format(obj),fontsize = 14)
            axs[no,obj-1].xaxis.set_tick_params(labelsize=11)
            axs[no,obj-1].yaxis.set_tick_params(labelsize=11)
handles = []
for i in range(4):
    handles.append(tuple(fs[i]))
fig.legend(handles,f_legend,fontsize = 14)
fig.savefig('fig3.pdf',dpi =300 ,bbox_inches='tight')