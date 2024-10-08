#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn, Tensor
from pycocotools.coco import COCO
import numpy as np
import pickle
import os
import torch
from matplotlib import pyplot as plt
from PIL import Image
import requests
from transformers import AutoProcessor, BlipForConditionalGeneration

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

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

if( not os.path.isdir('Features_BLIP/')):
    os.mkdir('Features_BLIP/')

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        for layer_id in layers:
            self.model.vision_model.encoder.layers[layer_id].register_forward_hook(self.save_outputs_hook(layer_id))
    def save_outputs_hook(self, layer_id): 
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn

    def forward(self, x):
        caption = self.model.generate(**x)
        return self._features,caption


layers = []
for l in range(0,12):
    layers.append(l)
image_base_dir = 'coco/images/train2017/'

feature_extractor= FeatureExtractor(model,layers)

for task in range(6):
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
                image = Image.open(img_file) # load image 
                inputs = processor(images=image, text=text, return_tensors="pt") # inputs for the model
                feature,caption = feature_extractor.forward(inputs)
                caption = processor.decode(caption[0], skip_special_tokens=True)
                img_dict= {}
                img_dict['id']= info['id']
                for ob in [obj1,obj2]:
                    cat_id = cat_name_to_id[ob]
                    mask = get_mask(coco_annotation,img_id,cat_id) # get mask from image
                    mask_img = Image.fromarray(mask) # convert to image
                    resized = np.array(mask_img.resize((24,24),Image.NEAREST)).flatten() # resize to size of tokens
                    sel_index = np.where(resized>0)[0]  # select object tokens index
                    if(len(sel_index)==0):
                        continue_flag=True
                        break 
                    img_dict[ob] = {}
                    img_dict[ob]['all_features']={}
                    img_dict[ob]['random_features']={}
                    img_dict[ob]['all_random']={}
                    img_dict[ob]['cls_features']={}
                    img_dict[ob]['caption']=caption
                    for layer in layers: # get the token for each layer
                        random_token = np.random.choice(sel_index)+1 # get a random token index from the object
                        random_full = np.random.choice(np.arange(0,576))+1 # get a random token index from the whole image
                        img_dict[ob]['all_features'][layer] = np.mean(feature[layer][0][0,sel_index+1].cpu().numpy(),axis=0) # average tokens from the object mask
                        img_dict[ob]['random_features'][layer] = feature[layer][0][0,random_token].cpu().numpy()
                        img_dict[ob]['cls_features'][layer] = feature[layer][0][0,0].cpu().numpy()
                        img_dict[ob]['all_random'][layer] = feature[layer][0][0,random_full].cpu().numpy()
                if(continue_flag):
                    continue
                all_features.append(img_dict)
                if((len(all_features)+1) % 100 == 0): # split data into smaller files
                    f = open('Features_BLIP/'+task_name+'_'+obj1+'_'+obj2+'_'+str(counter)+'.pkl','wb')
                    pickle.dump(all_features,f)
                    f.close()
                    counter += 1
                    all_features = []
            f = open('Features_BLIP/'+task_name+'_'+obj1+'_'+obj2+'_'+str(counter)+'.pkl','wb')
            pickle.dump(all_features,f)
            f.close()   
    
    
    
    
    
