import numpy as np
from PIL import Image 
import torch
import pickle

class FeatureExtractor(torch.nn.Module):
    def __init__(self, model, layers):
        super().__init__()
        self.model = model
        self.layers = layers
        self._features = {layer: torch.empty(0) for layer in layers}
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.model = self.model.to(self.device)

        for layer_id in layers:
            self.model.vision_model.encoder.layers[layer_id].register_forward_hook(self.save_outputs_hook(layer_id))
            
    def save_outputs_hook(self, layer_id): 
        def fn(_, __, output):
            self._features[layer_id] = output
        return fn
        
    def forward(self, x):
        x = x.to(self.device)
        caption = self.model.generate(**x)
        return self._features, caption
        
def generate_features(model,processor,feature_extractor,coco,image_annotations,input_folder, output_folder):
    layers = np.arange(0,12)
    images_metas = []
    for nums, id in enumerate(image_annotations):
        img_dict = {}
        # add image id 
        img_dict['id']= id['id']
        image_name = id['file_name'] # get image file_name
        
        # load image and get features
        image = Image.open(input_folder + image_name) 
        inputs = processor(images=image, text="", return_tensors="pt")
        feature,caption = feature_extractor.forward(inputs)
        
        annotations_ids = coco.getAnnIds(imgIds=id['id'])
        object_annotations = coco.loadAnns(annotations_ids)
        
        for l in layers:
            img_dict[l] = {}
            for annotation in object_annotations:
                
                
                category_id = annotation['category_id']
                
                img_dict[l][category_id] = {}
    
                # get object mask
                mask = coco.annToMask(annotation)  
                mask_img = Image.fromarray(mask) 
                
                # scale mask to token space
                resized = np.array(mask_img.resize((24,24),Image.NEAREST)).flatten()         
                
                # select object tokens index
                sel_index = np.where(resized>0)[0]  
                if(len(sel_index)==0):
                    continue_flag=True
                    break 
                    
                random_token = np.random.choice(sel_index)+1
                random_full = np.random.choice(np.arange(0,576))+1
                
                img_dict[l][category_id]['all_features'] = np.mean(feature[l][0][0,sel_index+1].cpu().numpy(),axis=0)
                img_dict[l][category_id]['random_features'] = feature[l][0][0,random_token].cpu().numpy()
    
            img_dict[l]['cls_features'] = feature[l][0][0,0].cpu().numpy()
            img_dict[l]['all_random'] = feature[l][0][0,random_full].cpu().numpy()
    
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