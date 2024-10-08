#!/usr/bin/env python
# coding: utf-8


from torch import nn, Tensor
from pycocotools.coco import COCO
import numpy as np
import pickle
import torch
from matplotlib import pyplot as plt
from PIL import Image
import requests

coco_annotation_file_path = "coco/images/annotations/instances_train2017.json"
coco_annotation = COCO(annotation_file=coco_annotation_file_path)


cat_name_to_id = {}
cat_ids = coco_annotation.getCatIds()
cats = coco_annotation.loadCats(cat_ids)
for ind,id in enumerate(cat_ids):
    cat_name_to_id[cats[ind]['name']] = cats[ind]['id']



def get_mask(coco, img_id, cat_id):
    '''function to get object mask for a given coco meta datas instance, image_id and cat_id: category id'''
    anns_ids= coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    mask = coco.annToMask(anns[0])
    for i in range(1,len(anns)):
        mask += coco.annToMask(anns[i])
    return mask



import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("RN50x4", device=device)


if( not os.path.isdir('Features_CNN/')):
    os.mkdir('Features_CNN/')

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        model.visual.layer1.register_forward_hook(self.save_outputs_hook(1))
        model.visual.layer2.register_forward_hook(self.save_outputs_hook(2))
        model.visual.layer3.register_forward_hook(self.save_outputs_hook(3))
        model.visual.layer4.register_forward_hook(self.save_outputs_hook(4))
            
    def save_outputs_hook(self, layer_id): 
        def fn(_, __, output):
            self._features[layer_id] = output[0].detach()
        return fn
        
    def forward(self, x):
        with torch.no_grad():
            f = model.encode_image(x)
        return self._features

layers = []
for l in range(1,5):
    layers.append(l)

feature_extractor= FeatureExtractor(model,layers)

image_base_dir = 'coco/images/train2017/'

for task in range(0,6):
    task_name = 'task_{}'.format(task)
     
    f = open('Tasks/'+task_name +'.pkl','rb')
    meta_by_objects= pickle.load(f)
    text = "" # initial text for geneation
    for obj1 in meta_by_objects.keys():
        for obj2 in meta_by_objects[obj1].keys():
            print(obj1,obj2)
            infos = coco_annotation.loadImgs(list(meta_by_objects[obj1][obj2]))
            all_features = []
            counter = 1
            for ind,info in enumerate(infos):
                continue_flag = False
                print(ind,'of',len(infos),len(all_features)) # tracking index
                img_file = image_base_dir + info['file_name'] # filename of image
                img_id = info['id'] # image id 
                image = np.array(Image.open(img_file)) # load image 
                image = preprocess(Image.fromarray(image)).unsqueeze(0).to(device)
                feature =feature_extractor.forward(image)
                img_dict= {}
                img_dict['id']= info['id']
                for ob in [obj1,obj2]:
                    cat_id = cat_name_to_id[ob]
                    mask = get_mask(coco_annotation,img_id,cat_id) # get mask from image
                    img_dict[ob] = {}
                    img_dict[ob]['all_features']= {}
                    img_dict[ob]['random_features']= {}
                    img_dict[ob]['all_random']= {}
                    img_dict[ob]['avg']= {}
                    for layer in layers:
                        mask_img = Image.fromarray(mask) # convert to image
                        dims = feature[layer].shape
                        resized = np.array(mask_img.resize((dims[1],dims[2]),Image.NEAREST)).flatten() # resize to size of tokens
                        sel_index = np.where(resized>0)[0]  # select object tokens index
                        if(len(sel_index)==0):
                            continue_flag=True
                            break 
                        random_token = np.random.choice(sel_index) # random token from object masked featuremap
                        random_full = np.random.choice(np.arange(0,dims[1]*dims[2])) # random cell from the whole image feature map
                        f_temp = feature[layer].cpu().numpy().reshape(dims[0],dims[1]*dims[2])
                        img_dict[ob]['all_features'][layer] = np.mean(f_temp[:,sel_index],axis=1) # average object representation
                        img_dict[ob]['avg'][layer] = np.mean(f_temp[:,:],axis=1) # average representations over the whole feature space
                        img_dict[ob]['random_features'][layer] = f_temp[:,random_token] 
                        img_dict[ob]['all_random'][layer] = f_temp[:,random_full]
                if(continue_flag):
                    continue
                all_features.append(img_dict)
                if((len(all_features)+1) % 50 == 0):
                    f = open('Features_CNN/'+task_name+'_'+obj1+'_'+obj2+'_'+str(counter)+'.pkl','wb')
                    pickle.dump(all_features,f)
                    f.close()
                    counter += 1
                    all_features = []
            f = open('Features_CNN/'+task_name+'_'+obj1+'_'+obj2+'_'+str(counter)+'.pkl','wb')
            pickle.dump(all_features,f)
            f.close()   
    
    
    
    
    
