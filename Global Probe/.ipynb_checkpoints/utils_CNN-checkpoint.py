import numpy as np
from PIL import Image 
import torch
import pickle
import torch
import clip
from PIL import Image


        
class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        self._features = {layer: torch.empty(0) for layer in layers}
        model.visual.layer1.register_forward_hook(self.save_outputs_hook(0))
        model.visual.layer2.register_forward_hook(self.save_outputs_hook(1))
        model.visual.layer3.register_forward_hook(self.save_outputs_hook(2))
        model.visual.layer4.register_forward_hook(self.save_outputs_hook(3))
            
    def save_outputs_hook(self, layer_id): 
        def fn(_, __, output):
            self._features[layer_id] = output[0].detach()
        return fn
        
    def forward(self, x):
        x = x.to(self.device)
        with torch.no_grad():
            f = self.model.encode_image(x)
        return self._features     
        
def generate_features(model,processor,feature_extractor,coco,image_annotations,input_folder, output_folder):
    layers = np.arange(4)
    images_metas = []
    for nums, id in enumerate(image_annotations):
        img_dict = {}
        # add image id
        img_dict['id']= id['id']
        image_name = id['file_name'] # get image file_name
        
        # load image and get features
        image = np.array(Image.open(input_folder + image_name)) # load image 
        inputs = processor(Image.fromarray(image)).unsqueeze(0)
        feature = feature_extractor.forward(inputs)
        
        annotations_ids = coco.getAnnIds(imgIds=id['id'])
        object_annotations = coco.loadAnns(annotations_ids)
        
        for l in layers:
            img_dict[l] = {}
            dims = feature[l].shape
            f_temp = feature[l].cpu().numpy().reshape(dims[0],dims[1]*dims[2])
            for annotation in object_annotations:
                
                
                category_id = annotation['category_id']
                
                img_dict[l][category_id] = {}
    
                # get object mask
                mask = coco.annToMask(annotation)  
                mask_img = Image.fromarray(mask) 
                
                # scale mask to token space
                
                resized = np.array(mask_img.resize((dims[1],dims[2]),Image.NEAREST)).flatten() # resize to size of tokens
                
                # select object tokens index
                sel_index = np.where(resized>0)[0]  
                if(len(sel_index)==0):
                    continue_flag=True
                    break 
                    
                random_token = np.random.choice(sel_index)
                
                img_dict[l][category_id]['all_features'] = np.mean(f_temp[:,sel_index],axis=1)
                img_dict[l][category_id]['random_features'] = f_temp[:,random_token]
                
            random_full = np.random.choice(np.arange(0,dims[1]*dims[2]))
            img_dict[l]['cls_features'] = np.mean(f_temp[:,:],axis=1)
            img_dict[l]['all_random'] = f_temp[:,random_full]
    
        # Append the image dictionary to the list
        images_metas.append(img_dict)
    
        if(len(images_metas) % 50 == 0):
            with open(f'{output_folder}/features_batch_{int(nums/50)}','wb') as f:
                pickle.dump(images_metas,f)
            images_metas = []
        print(nums)
        
    # save remaining elements
    if(images_metas):
        with open(f'{output_folder}/features_batch_{int(nums/50)}','wb') as f:
            pickle.dump(images_metas,f)