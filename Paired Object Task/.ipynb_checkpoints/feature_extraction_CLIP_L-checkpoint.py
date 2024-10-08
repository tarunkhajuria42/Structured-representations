#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torch import nn, Tensor
from pycocotools.coco import COCO
import numpy as np
import pickle
import torch
from matplotlib import pyplot as plt
from PIL import Image
import requests
from transformers import CLIPProcessor, CLIPModel

# In[2]:


coco_annotation_file_path = "../../../../coco/images/annotations/instances_train2017.json"
coco_annotation = COCO(annotation_file=coco_annotation_file_path)


cat_name_to_id = {}
cat_ids = coco_annotation.getCatIds()
cats = coco_annotation.loadCats(cat_ids)
for ind,id in enumerate(cat_ids):
    cat_name_to_id[cats[ind]['name']] = cats[ind]['id']



def get_mask(coco, img_id, cat_id):
    anns_ids= coco.getAnnIds(imgIds=img_id, catIds=cat_id, iscrowd=None)
    anns = coco.loadAnns(anns_ids)
    mask = coco.annToMask(anns[0])
    for i in range(1,len(anns)):
        mask += coco.annToMask(anns[i])
    return mask



model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")


class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        self._attention = {}
        for layer_id in layers:
            self.model.vision_model.encoder.layers[layer_id].register_forward_hook(self.save_outputs_hook(layer_id))
    def save_outputs_hook(self, layer_id): 
        def fn(_, __, output):
            self._features[layer_id] = output[0].detach()
        return fn
    def forward(self, x):
        _ = self.model(**x)
        return self._features

for task in range(0,6):
    task_name = 'task_{}'.format(task)
     
    f = open('Tasks/'+task_name +'.pkl','rb')
    meta_by_objects= pickle.load(f)
    
    layers = []
    for l in range(0,24):
        layers.append(l)
    image_base_dir = '../../../../coco/images/train2017/'
    
    
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
                feature =FeatureExtractor(model,layers).forward(inputs)
                img_dict= {}
                img_dict['id']= info['id']
                for ob in [obj1,obj2]:
                    cat_id = cat_name_to_id[ob]
                    mask = get_mask(coco_annotation,img_id,cat_id) # get mask from image
                    mask_img = Image.fromarray(mask) # convert to image
                    resized = np.array(mask_img.resize((14,14),Image.NEAREST)).flatten() # resize to size of tokens
                    sel_index = np.where(resized>0)[0]  # select object tokens index
                    if(len(sel_index)==0):
                        continue_flag=True
                        break 
                    img_dict[ob] = {}
                    img_dict[ob]['all_features']={}
                    img_dict[ob]['random_features']={}
                    img_dict[ob]['all_random']={}
                    img_dict[ob]['cls_features']={}
                    #img_dict[ob]['norms']={}
                    for layer in layers:
                        random_token = np.random.choice(sel_index)+1
                        random_full = np.random.choice(np.arange(0,196))+1
                        #norms = np.linalg.norm(feature[layer][0][:],axis=1)
                        img_dict[ob]['all_features'][layer] = np.mean(feature[layer][0][sel_index+1].cpu().numpy(),axis=0)
                        img_dict[ob]['random_features'][layer] = feature[layer][0][random_token].cpu().numpy()
                        img_dict[ob]['cls_features'][layer] = feature[layer][0][0].cpu().numpy()
                        img_dict[ob]['all_random'][layer] = feature[layer][0][random_full].cpu().numpy()
                        #img_dict[ob]['norms'][layer] = feature[layer][0][np.argsort(norms)[-3:]].cpu().numpy()
                if(continue_flag):
                    continue_flag = False
                    continue
                all_features.append(img_dict)
                if((len(all_features)+1) % 100 == 0):
                    f = open('Features_CLIP_Large/'+task_name+'_'+obj1+'_'+obj2+'_'+str(counter)+'.pkl','wb')
                    pickle.dump(all_features,f)
                    f.close()
                    counter += 1
                    all_features = []
            f = open('Features_CLIP_Large/'+task_name+'_'+obj1+'_'+obj2+'_'+str(counter)+'.pkl','wb')
            pickle.dump(all_features,f)
            f.close()   
    
    
    
    
    
