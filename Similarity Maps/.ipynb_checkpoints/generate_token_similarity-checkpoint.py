import os
import json
import pickle
import copy
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns

from feature_extraction import get_objects_mask, extract_tokens
from blip_analysis import find_original_img_patch
from blip_analysis import BlipCaptioningModel
from clip_analysis import ClipCaptioningModel
from flava_analysis import FlavaCaptioningModel

os.environ["WANDB_DISABLED"] = "true"

COCO_PATH = "../../../datasets/coco"
DATA_SPLIT = "train2017"

# for diplaying segmentation mask
def add_annotations(img: Image, annotations: list, skip_objects=[], fill=(255,0,255,150), outline=(0,0,0,255)):
    for ann in annotations:
        if isinstance(ann['segmentation'], list) and not ann["id"] in skip_objects:
            seg = ann['segmentation'][0]
            poly = Image.new('RGBA', img.size)
            img_draw = ImageDraw.Draw(poly)  
            img_draw.polygon(ann['segmentation'][0],fill=fill, outline=outline)
            img.paste(poly,mask=poly)
    return img

def get_all_object_masks(img, annotations):
    masks = []
  
    for ann in annotations['annotations']['annotations']:
        pil_mask = np.zeros(shape=img.size, dtype=np.uint8)
        pil_mask = Image.fromarray(np.moveaxis(pil_mask, 0, -1))
        img_draw = ImageDraw.Draw(pil_mask) 
        
        segment = ann["segmentation"]
        if isinstance(segment, list):
            try:
                img_draw.polygon(segment[0], fill ="#ffffff")
            except ValueError:
                return None
        
        masks.append(pil_mask)
    
    return masks

def token_similarity(base_embedding:list, hidden_states, layers:list, n_tokens=196):
    similarity = np.zeros(shape=(len(layers),n_tokens))
    for layer_idx, (embedding, layer) in enumerate(zip(base_embedding, layers)):
        for token in range(1,n_tokens):
            # TODO: make it batch input based
            similarity[layer_idx][token-1] = cosine_similarity(embedding.reshape(1, -1), hidden_states[layer][0][token].reshape(1, -1))
    return similarity


def find_similar_image_tokens(object_data, token_type:str, layers_act, layers:list, n_tokens=196, threshold=0.33):
    similarities = np.zeros((len(layers),n_tokens))
    object_similarity_layer = np.zeros((len(object_data['class']),len(layers),n_tokens))
    most_similar_class = np.zeros((len(layers),n_tokens),dtype="int")

    for obj_idx, (obj, obj_class) in enumerate(zip(object_data['object_tokens_embedding'], object_data['class'])):
        embeddings = [emb[token_type] for layer, emb in obj.items()]
        obj_token_similarity = token_similarity(embeddings, layers_act, layers=layers, n_tokens=n_tokens)
        object_similarity_layer[obj_idx] = copy.deepcopy(obj_token_similarity)
        for layer,_ in enumerate(layers):
            for token_idx, (obj_a, obj_b) in enumerate(zip(similarities[layer], obj_token_similarity[layer])):
                if obj_b > obj_a and obj_b > threshold:
                    similarities[layer][token_idx] = obj_b
                    most_similar_class[layer][token_idx] = obj_class    
    return most_similar_class, similarities, object_similarity_layer
    
def generate_mask_overlay(img, objects_data, most_similar_class, similarities, token_type, img_size:tuple, patch_size:int, grid_size:int):
    # create overlay image corresponding to token regions
    img_overlays = []
    similarity_masks = []
    
    font = ImageFont.truetype("/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf", 14)
    # TODO: increase color limit to 10 classes, currently 5 + no class
    colors = ["black","yellow","green","blue"] #"grey","purple","orange","green","blue","grey","purple"
    classes = [0] + list(set(objects_data["class"])) + [999]
    classes = sorted(classes)
    color_class_map = {class_:color for color,class_ in zip(colors, classes)}

    # create a 16x16 imag patch with the color of dominant class of the patch
    # returning a 196-array with class-colored patches
    # most_similar_class is layer-based (n-layer, 196-tokens)
    for layer_idx, layer_similarity in enumerate(most_similar_class): 
        layer_class_overlays = []
        for patch_idx, patch_class in enumerate(layer_similarity):
            token_overlay = Image.new(mode="RGBA", size=(patch_size, patch_size), color=color_class_map[patch_class])
            layer_class_overlays.append(token_overlay)
            
        # create a standard input (224x224) image to reconstruct the patch space (14x14 grid of pathces/tokens)
        similarity_mask = Image.new(mode="RGBA", size=img_size, color="white")
        img_patch = 0
        for i in range(0,img_size[0], patch_size):
            for j in range(0,img_size[1], patch_size):
                similarity_mask.paste(layer_class_overlays[img_patch], (j,i))
                img_patch += 1
        similarity_masks.append(similarity_mask)
        overlay = Image.blend(img.resize(img_size).convert("RGBA"), similarity_mask.convert("RGBA"), 0.5)
        img_overlays.append(copy.deepcopy(overlay))
        
    # Draw symbol to mark origin token for each object
    for object_tokens in objects_data["object_tokens"]:
        #each layer
        layer_idx = 0
        for layer, highlight_token in object_tokens.items():
            token = highlight_token[token_type]            
            coord = find_original_img_patch(token, np.array(img.resize(img_size)), grid_size=grid_size, patch_size=patch_size)[0]
            img_draw = ImageDraw.Draw(img_overlays[layer_idx])     
            # print(f"writting token at {layer_idx} token {token} ({token_type})")                   
            img_draw.text((coord[2]+4,coord[0]+4), "*", fill ="red", font = font, align ="right", anchor="lt") 
            layer_idx += 1
       
    return img_overlays, similarity_masks, color_class_map


def get_bg_mask(fg_mask:np.array):
    # get background mask by reversing the image mask
    bg_mask_img = copy.deepcopy(fg_mask)
    bg_mask_img[fg_mask>0] = 0
    bg_mask_img[fg_mask<=0] = 255
    return Image.fromarray(bg_mask_img)

def get_model_config(model_name):
    if model_name=="BLIP":
        img_size, patch_size, grid_size, n_tokens = (384,384), 16, 24, 576
    elif model_name=="BLIP-L":
        img_size, patch_size, grid_size, n_tokens = (384,384), 16, 24, 576
    elif model_name =="CLIP":
        img_size, patch_size, grid_size, n_tokens = (224,224), 16, 14, 196
    elif model_name =="CLIP-L":
        img_size, patch_size, grid_size, n_tokens = (224,224), 14, 16, 256
    elif model_name =="FLAVA":
        img_size, patch_size, grid_size, n_tokens = (224,224), 16, 14, 196        
    else:
        raise
    
    return img_size, patch_size, grid_size, n_tokens

def generate_similarity_plot(input_annotation, 
                             model, 
                             object_pair = None, 
                             layers = [9,10,11], 
                             token_type="random_obj",
                             stable_obj_tokens=None, 
                             data_split="val2017", 
                             model_arc="CLIP"):        
    
    img_size, patch_size, grid_size, n_tokens = get_model_config(model_arc)
    
    if isinstance(object_pair, tuple):
        object_pair = [int(object_pair[0]), int(object_pair[1])]
    img = Image.open(os.path.join(COCO_PATH, data_split, input_annotation['image']['file_name'])).convert('RGB')        
    outputs = model.forward_pass([img])
    if model_arc=="FLAVA":
        layers_act = torch.stack(outputs.image_output.hidden_states).cpu().detach().numpy()
        mean_att_map = model.get_all_attention_maps(torch.stack(outputs.image_output.attentions), renorm_weights=True)         
    else:
        layers_act = torch.stack(outputs.vision_model_output.hidden_states).cpu().detach().numpy()
        mean_att_map = model.get_all_attention_maps(torch.stack(outputs.vision_model_output.attentions), renorm_weights=True)            
    mean_att_map = mean_att_map.cpu().detach().numpy()
    
    # TODO: Make it select only primary,secondary objects
    # get object masks for all objects
    # all_masks = get_all_object_masks(img, input_annotation)
    all_obj_masks, obj_ann = get_objects_mask(annotation=input_annotation, 
                                              objects=object_pair, 
                                              option=1, 
                                              attention_maps=mean_att_map[:,0,:,:],
                                              layers=layers,
                                              split=data_split)  
    
    # Generate mask including both foreground objects, for later getting the background mask based on these 2 objects
    # hardcoded for 2 objects
    fg_masks = np.array(all_obj_masks[0][0]) + np.array(all_obj_masks[1][0]) 
    bg_masks = [[get_bg_mask(fg_masks)]]    
    
    # extract relevant tokens and layer embeddings
    objects_data = run_token_extraction(all_obj_masks, 
                                        bg_masks, 
                                        img, 
                                        obj_ann, 
                                        layers, 
                                        mean_att_map, 
                                        layers_act, 
                                        input_annotation['image']['id'], 
                                        input_annotation['image']['file_name'],
                                        stable_tokens=stable_obj_tokens,
                                        img_size=img_size,
                                        patch_size=patch_size,
                                        grid_size=grid_size,
                                        n_tokens=n_tokens                                        
                                        )
    
    ## overrides token per layer to make it consistent across all layers. Choose one token from the object (last layer) and keep it for all layers 
    ## for that object
    for obj_idx, obj_fg_tokens in enumerate(objects_data["object_tokens"]):
        # get last layer tokens
        min_obj_stable = obj_fg_tokens[layers[-1]]["min_obj"]
        max_obj_stable = obj_fg_tokens[layers[-1]]["max_obj"]
        random_obj_stable = obj_fg_tokens[layers[-1]]["random_obj"]
        max_image_stable = obj_fg_tokens[layers[-1]]["max_image"]
        print(f"Stable token {random_obj_stable}")
        for layer, tokens in objects_data["object_tokens_embedding"][obj_idx].items():
            objects_data["object_tokens"][obj_idx][layer]["min_obj"] = min_obj_stable
            objects_data["object_tokens"][obj_idx][layer]["max_obj"] = max_obj_stable
            objects_data["object_tokens"][obj_idx][layer]["random_obj"] = random_obj_stable
            objects_data["object_tokens"][obj_idx][layer]["max_image"] = max_image_stable
            objects_data["object_tokens_embedding"][obj_idx][layer] = {'min_obj':    layers_act[layer][0][min_obj_stable+1],
                                                                       'max_obj':   layers_act[layer][0][max_obj_stable+1],
                                                                       'random_obj':layers_act[layer][0][random_obj_stable+1],
                                                                       'max_image': layers_act[layer][0][max_image_stable+1]}
                                        
    # find similar tokens for each object, for each layer based on a reference embedding (aka selected token)
    most_similar_class, similarities, obj_token_similarity = find_similar_image_tokens(objects_data, token_type, layers_act, layers, n_tokens=n_tokens)
    
    # print(similarities.shape)
    # for layer in range(len(layers)):
    #     plt.imshow(similarities[layer].reshape((14,14)))
    #     plt.show()
    
    # finally generate image overlays with token similarities
    img_overlays, similarity_masks, color_class_map = generate_mask_overlay(img, 
                                                                            objects_data, 
                                                                            most_similar_class, 
                                                                            similarities, 
                                                                            token_type=token_type,
                                                                            img_size=img_size,
                                                                            patch_size=patch_size,
                                                                            grid_size=grid_size)
    
    selected_tokens = {obj_idx: obj_token[layers[-1]][token_type] for obj_idx, obj_token in enumerate(objects_data["object_tokens"])}
    
    return img_overlays, similarity_masks, obj_token_similarity, color_class_map, selected_tokens, all_obj_masks
    
    
def run_token_extraction(all_masks, 
                         bg_mask, img: Image, 
                         annotation, 
                         layers:list, 
                         mean_att_map, 
                         layers_act, 
                         image_id=None, 
                         image_filename=None,
                         stable_tokens=None,
                         img_size=(224, 224),
                         grid_size=14,
                         patch_size=16,
                         n_tokens=196):
    data = {"image_id": [],
            "image_filename": [],
            "caption_filter": [], 
            "object_tokens": [],
            "object_tokens_embedding": [],
            "is_background": [],
            "class": []}
    
    def insert_token_data(token_data, image_id, image_filename, tokens, layers_act, object_class, is_background ):
        token_data["image_id"].append(image_id)
        token_data["image_filename"].append(image_filename)
        token_data["object_tokens"].append(tokens)
        token_act_per_layer = {}
        for layer, token in tokens.items():
            token_act_per_layer[layer] = {'min_obj': layers_act[layer][0][token['min_obj']+1],
                                          'max_obj': layers_act[layer][0][token['max_obj']+1],
                                          'random_obj': layers_act[layer][0][token['random_obj']+1],
                                          'max_image': layers_act[layer][0][token['max_image']+1]}

        token_data["object_tokens_embedding"].append(token_act_per_layer)                    
        # token_data["caption_filter"].append(batch_ann[input_i].get('annotations').get('objects_in_caption', False))
        token_data["class"].append(object_class)    # layer 0, bactch input 0?
        token_data["is_background"].append(is_background)    
        
        return token_data   
    
    # IF tokens are known, just assign the data
    # This way we can assign a set of tokens and extract data for multiple architectures and compare them
    if stable_tokens is not None:
        # for obj_idx, obj in enumerate(annotation):
        for obj_idx, token_no in stable_tokens.items():
            if obj_idx < len(annotation)-1:
                obj_class = int(annotation[obj_idx][0]["category_id"])
                is_background = False
            else:
                obj_class = 999 #background
                is_background = True
            obj_tokens = {}
            for layer_no in layers: 
                obj_tokens[layer_no] = {'min_obj': token_no,
                                        'max_obj': token_no,
                                        'random_obj': token_no,
                                        'max_image': token_no}
            data = insert_token_data(data, image_id, image_filename, obj_tokens, layers_act, object_class=obj_class,is_background=is_background)     
            
        return data

    # if tokens are not known, find min, max, random, random img tokens based on attention maps
    # 1 Extract token(s) from objects in the scene
    for mask, ann in zip(all_masks, annotation):
        # OBJECT
        # repeat the same object mask per layer to keep compatibility (all layers have the same object mask)
        object_mask = len(layers) * [np.array(mask[0].resize(img_size))]
        # for layer based mask (different mask per layer)
        # object_mask = [np.array(m.resize((224,224))) for m in mask]
        obj_fg_tokens, _ = extract_tokens(img=img.resize(img_size), 
                                                        object_mask=object_mask,
                                                        mean_att_map=mean_att_map[:,0,:,:],
                                                        layers=layers,
                                                        grid_size=grid_size,
                                                        patch_size=patch_size,
                                                        n_tokens=n_tokens,
                                                        mask_threshold=0)

        if obj_fg_tokens is not None: 
            data = insert_token_data(data, image_id,image_filename,obj_fg_tokens, layers_act, object_class=int(ann[0]["category_id"]),is_background=False)       
        else:
            print(f"Object from annotation {ann} didnt return any token: {obj_fg_tokens}")
            # plt.imshow(mask)
            # plt.show()
    # run for background
    # repeat the same object mask per layer to keep compatibility (all layers have the same object mask)
    bg_mask = len(layers) * [np.array(bg_mask[0][0].resize(img_size))]
    bg_tokens, _ = extract_tokens(img=img.resize(img_size), 
                                    object_mask=bg_mask,
                                    mean_att_map=mean_att_map[:,0,:,:],
                                    layers=layers,
                                    grid_size=grid_size,
                                    patch_size=patch_size,
                                    n_tokens=n_tokens,
                                    mask_threshold=0)
    if bg_tokens is not None: 
        data = insert_token_data(data, image_id, image_filename, bg_tokens, layers_act, object_class=999,is_background=True)     
    else:
        print(f"Background didnt return any token: {obj_fg_tokens}")
    
    return data

def plot_default_similarity_maps(models:list,                                 
                                 similarity_obj_matrix:tuple, 
                                 selected_tokens:tuple, 
                                 source_img_file_name, 
                                 data_split, 
                                 layers=[0,3,7,11], 
                                 savefile=""):
    
    sns.color_palette("rocket", as_cmap=True)
    fig, axs = plt.subplots(ncols=8, nrows=len(layers)+1, gridspec_kw=dict(width_ratios=[20,20,20,1.5,20,20,20,2]), figsize=(21,3.8*(len(layers)+1)))
    
    for j in range(0,8):
        axs[0, j].tick_params(left=False,bottom=False)
        axs[0, j].axes.xaxis.set_ticks([])
        axs[0, j].axes.yaxis.set_ticks([])
        axs[0, j].grid(visible=True)
        axs[0, j].axis("off")

    source_img = Image.open(os.path.join(COCO_PATH, data_split, source_img_file_name)).convert('RGB').resize((384,384))  
    img_draw = ImageDraw.Draw(source_img) 
    coord = find_original_img_patch(selected_tokens[0][0], np.array(source_img), grid_size=24, patch_size=16)[0]
    img_draw.rectangle([(coord[2],coord[0]),(coord[3],coord[1])], fill ="#ffff33", outline ="red")            

    axs[0, 1].set_title("Primary\nobject", size=35)
    axs[0, 1].imshow(source_img)
    axs[0, 1].tick_params(left=False,bottom=False)
    axs[0, 1].axes.xaxis.set_ticks([])
    axs[0, 1].axes.yaxis.set_ticks([])
    axs[0, 1].set_ylabel(str(f"Input"), size=40)

    source_img = Image.open(os.path.join(COCO_PATH, data_split, source_img_file_name)).convert('RGB').resize((384,384))  
    img_draw = ImageDraw.Draw(source_img) 
    coord = find_original_img_patch(selected_tokens[0][1], np.array(source_img), grid_size=24, patch_size=16)[0]
    img_draw.rectangle([(coord[2],coord[0]),(coord[3],coord[1])], fill ="#0000ff", outline ="red")            

    axs[0, 5].set_title("Secondary\nobject", size=35)
    axs[0, 5].imshow(source_img)
    axs[0, 5].tick_params(left=False,bottom=False)
    axs[0, 5].axes.xaxis.set_ticks([])
    axs[0, 5].axes.yaxis.set_ticks([])        
    # sns.set(font_scale=1)

    col_colorbar = 7
    col_space = 3
    for layer_idx, layer in enumerate(layers):            
        col = 0
        for object in [0,1]: 
            for model_idx, model in enumerate(models):
                img_size, patch_size, grid_size, n_token = get_model_config(model)
                # add a space between primary secondary objects
                if col == col_space:
                    axs[layer_idx+1, col].tick_params(left=False,bottom=False)
                    axs[layer_idx+1, col].axes.xaxis.set_ticks([])
                    axs[layer_idx+1, col].axes.yaxis.set_ticks([])
                    axs[layer_idx+1, col].grid(visible=False)     
                    axs[layer_idx+1, col].axis("off")
                    col += 1

                if col < col_colorbar-1:       
                    sns.heatmap(similarity_obj_matrix[model_idx][object,layer_idx,:].reshape((grid_size,grid_size)), ax=axs[layer_idx+1, col], cbar=False ,yticklabels=False, xticklabels=False, vmin=0, vmax=1)
                else:
                    # last col add color bar.
                    axsns = sns.heatmap(similarity_obj_matrix[model_idx][object,layer_idx,:].reshape((grid_size,grid_size)), ax=axs[layer_idx+1, col], cbar=True, cbar_ax=axs[layer_idx+1, col_colorbar], yticklabels=False, xticklabels=False, vmin=0, vmax=1)
                    cbar = axsns.collections[0].colorbar
                    # here set the labelsize by 20
                    cbar.ax.tick_params(labelsize=20)
                if layer_idx == 0 :
                    axs[layer_idx+1, col].set_title(model, size=35)   
                col += 1
        axs[layer_idx+1,0].set_ylabel(str(f"Layer {layer+1}"),size=40)
    # fig.colorbar(axs[layer_idx+1, 2].collections[layer_idx+1], cax=axs[layer_idx+1, 2])
    # fig.colorbar(axs[0, 2].collections[0], cax=axs[0, 2])
    plt.tight_layout()
    if savefile:
        plt.savefig(fname=savefile, dpi=500)
        
def plot_large_similarity_maps(models:list,                                 
                               similarity_obj_matrix:tuple, 
                               selected_tokens:tuple, 
                               source_img_file_name, 
                               data_split, 
                               layers=[0,3,7,11], 
                               savefile="",
                               input_image_size=(384,384),
                               input_image_grid_size=24,
                               input_image_patch_size=16):
    
    sns.color_palette("rocket", as_cmap=True)
    fig, axs = plt.subplots(ncols=6, nrows=len(layers)+1, gridspec_kw=dict(width_ratios=[20,20,1.5,20,20,2]), figsize=(12.5,3*(len(layers)+1)))
    
    for j in range(0,6):
        axs[0, j].tick_params(left=False,bottom=False)
        axs[0, j].axes.xaxis.set_ticks([])
        axs[0, j].axes.yaxis.set_ticks([])
        axs[0, j].grid(visible=True)
        axs[0, j].axis("off")

    source_img = Image.open(os.path.join(COCO_PATH, data_split, source_img_file_name)).convert('RGB').resize(input_image_size)  
    img_draw = ImageDraw.Draw(source_img) 
    coord = find_original_img_patch(selected_tokens[0][0], 
                                    np.array(source_img), 
                                    grid_size=input_image_grid_size, 
                                    patch_size=input_image_patch_size)[0]
    img_draw.rectangle([(coord[2],coord[0]),(coord[3],coord[1])], fill ="#ffff33", outline ="red")            

    axs[0, 1].set_title("Primary\nobject", size=35)
    axs[0, 1].imshow(source_img)
    axs[0, 1].tick_params(left=False,bottom=False)
    axs[0, 1].axes.xaxis.set_ticks([])
    axs[0, 1].axes.yaxis.set_ticks([])
    axs[0, 1].set_ylabel(str(f"Input"), size=40)

    source_img = Image.open(os.path.join(COCO_PATH, data_split, source_img_file_name)).convert('RGB').resize(input_image_size)  
    img_draw = ImageDraw.Draw(source_img) 
    coord = find_original_img_patch(selected_tokens[0][1], 
                                    np.array(source_img), 
                                    grid_size=input_image_grid_size, 
                                    patch_size=input_image_patch_size)[0]
    img_draw.rectangle([(coord[2],coord[0]),(coord[3],coord[1])], fill ="#0000ff", outline ="red")            

    axs[0, 3].set_title("Secondary\nobject", size=35)
    axs[0, 3].imshow(source_img)
    axs[0, 3].tick_params(left=False,bottom=False)
    axs[0, 3].axes.xaxis.set_ticks([])
    axs[0, 3].axes.yaxis.set_ticks([])        
    # sns.set(font_scale=1)

    col_colorbar = 5
    col_space = 2
    for layer_idx, layer in enumerate(layers):            
        col = 0
        for object in [0,1]: 
            for model_idx, model in enumerate(models):
                img_size, patch_size, grid_size, n_token = get_model_config(model)
                # add a space between primary secondary objects
                if col == col_space:
                    axs[layer_idx+1, col].tick_params(left=False,bottom=False)
                    axs[layer_idx+1, col].axes.xaxis.set_ticks([])
                    axs[layer_idx+1, col].axes.yaxis.set_ticks([])
                    axs[layer_idx+1, col].grid(visible=False)     
                    axs[layer_idx+1, col].axis("off")
                    col += 1

                if col < col_colorbar-1:       
                    sns.heatmap(similarity_obj_matrix[model_idx][object,layer_idx,:].reshape((grid_size,grid_size)), ax=axs[layer_idx+1, col], cbar=False ,yticklabels=False, xticklabels=False, vmin=0, vmax=1)
                else:
                    # last col add color bar.
                    axsns = sns.heatmap(similarity_obj_matrix[model_idx][object,layer_idx,:].reshape((grid_size,grid_size)), ax=axs[layer_idx+1, col], cbar=True, cbar_ax=axs[layer_idx+1, col_colorbar], yticklabels=False, xticklabels=False, vmin=0, vmax=1)
                    cbar = axsns.collections[0].colorbar
                    # here set the labelsize by 20
                    cbar.ax.tick_params(labelsize=20)
                if layer_idx == 0 :
                    axs[layer_idx+1, col].set_title(model, size=35)   
                col += 1
        axs[layer_idx+1,0].set_ylabel(str(f"Layer {layer+1}"),size=40)
    # fig.colorbar(axs[layer_idx+1, 2].collections[layer_idx+1], cax=axs[layer_idx+1, 2])
    # fig.colorbar(axs[0, 2].collections[0], cax=axs[0, 2])
    plt.tight_layout()
    if savefile:
        plt.savefig(fname=savefile, dpi=500)

def translate_token(source_token, source_img_size:int, source_grid_size, source_patch_size, dest_img_size:int, dest_grid_size, dest_patch_size):
    source_coord = find_original_img_patch(source_token, grid_size=source_grid_size, patch_size=source_patch_size)[0]
    # find the central point of the token, coord given is the (0,0) point
    x0, y0 = source_coord[0]+(source_patch_size/2), source_coord[2]+(source_patch_size/2)
    Rxy = dest_img_size/source_img_size
    x1, y1 = round(Rxy * x0), round(Rxy * y0)
    
    # project new coord into grid of destination image
    grd_x1 = x1 // dest_patch_size
    grd_y1 = y1 // dest_patch_size
    
    return (grd_x1 * dest_grid_size) + grd_y1


def load_sample_data(coco_image_id, data_split):
    with open(os.path.join(COCO_PATH, f"annotations/instances_{data_split}.json"), 'r') as f:
        coco_segmentation = json.load(f)
        
    sample_img = {}
    sample_img["image"] = [image for image in coco_segmentation["images"] if image["id"] == coco_image_id][0]
    sample_img["annotations"] = {"annotations": []}
    for annotation in coco_segmentation["annotations"]:
        if annotation["image_id"] == coco_image_id:
            sample_img["annotations"]["annotations"].append(annotation)
            
    return sample_img

def gen_model_data(input_image_data, model_obj, model_name, layers=[0,3,7,11], input_tokens=None, data_split="train2017"):
    similarity_plot, _, similarity_obj, _, selected_tokens,_ = generate_similarity_plot(input_annotation=input_image_data,
                                                                                        object_pair=(18,65),
                                                                                        model=model_obj,
                                                                                        token_type="random_obj",
                                                                                        layers=layers,
                                                                                        data_split=data_split,
                                                                                        model_arc=model_name,
                                                                                        stable_obj_tokens=input_tokens)
    
    with open(f"{model_name}_similarity_map.pickle", "wb") as output_file:
        pickle.dump({"similarities": similarity_obj, "tokens":selected_tokens}, output_file)
        
    return similarity_obj, selected_tokens

def main():
    input_image_data = load_sample_data(203371, DATA_SPLIT)
    
    stable_object_tokens = {0: 295, 1: 384, 2: 572}
    # translate tokens from BLIP space (24x24 grid) to CLIP/FLAVA (14x14 grid)
    t_stable_object_tokens = {obj:translate_token(src_token, 384, 24, 16, 224, 14, 16) for obj, src_token in stable_object_tokens.items()}
    layers = [0,2,4,6,8,11]
    layers_large = [0,4,8,12,16,23]
    
    model = BlipCaptioningModel(output_encoder_attentions=True, output_encoder_hidden_states=True)  
    blip_similarity_obj, blip_selected_tokens = gen_model_data(input_image_data,
                                                     model,
                                                     "BLIP",
                                                     layers=layers,
                                                     input_tokens=stable_object_tokens,
                                                     data_split=DATA_SPLIT)    
   
    model = ClipCaptioningModel(output_encoder_attentions=True, output_encoder_hidden_states=True)  
    clip_similarity_obj, clip_selected_tokens = gen_model_data(input_image_data,
                                                     model,
                                                     "CLIP",
                                                     layers=layers,
                                                     input_tokens=t_stable_object_tokens,
                                                     data_split=DATA_SPLIT)
    
    model = FlavaCaptioningModel(output_encoder_attentions=True, output_encoder_hidden_states=True)
    flava_similarity_obj, flava_selected_tokens = gen_model_data(input_image_data,
                                                     model,
                                                     "DINO",
                                                     layers=layers,
                                                     input_tokens=t_stable_object_tokens,
                                                     data_split=DATA_SPLIT)
    
    plot_default_similarity_maps(["BLIP", "CLIP", "DINO"], 
                                (blip_similarity_obj, clip_similarity_obj, flava_similarity_obj),
                                (blip_selected_tokens, clip_selected_tokens, flava_selected_tokens),
                                input_image_data["image"]['file_name'], 
                                DATA_SPLIT,
                                layers=layers,
                                savefile=f"fig_base_models_similarity_{str(len(layers))}_layers.png")
    
    model = BlipCaptioningModel(pretrained_model="Salesforce/blip-image-captioning-large", 
                                output_encoder_attentions=True, 
                                output_encoder_hidden_states=True)      
    blip_l_similarity_obj, blip_l_selected_tokens = gen_model_data(input_image_data,
                                                     model,
                                                     "BLIP-L",
                                                     layers=layers_large,
                                                     input_tokens=stable_object_tokens,
                                                     data_split=DATA_SPLIT)
    
    # CLIP-L has another configuration with patch_size 14
    t_stable_object_tokens = {obj:translate_token(src_token, 384, 24, 16, 224, 16, 14) for obj, src_token in stable_object_tokens.items()}
    model = ClipCaptioningModel(pretrained_model="openai/clip-vit-large-patch14", 
                                output_encoder_attentions=True, 
                                output_encoder_hidden_states=True)      
    clip_l_similarity_obj, clip_l_selected_tokens = gen_model_data(input_image_data,
                                                     model,
                                                     "CLIP-L",
                                                     layers=layers_large,
                                                     input_tokens=t_stable_object_tokens,
                                                     data_split=DATA_SPLIT)
    plot_large_similarity_maps(["BLIP-L", "CLIP-L"], 
                                (blip_l_similarity_obj, clip_l_similarity_obj),
                                (blip_l_selected_tokens, clip_l_selected_tokens),
                                input_image_data["image"]['file_name'], 
                                DATA_SPLIT,
                                layers=layers_large,
                                savefile=f"fig_large_models_similarity_{str(len(layers))}_layers.png")
    

if __name__ == "__main__":
    main()