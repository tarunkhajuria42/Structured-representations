import os
import random
import copy
from PIL import Image, ImageDraw
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

from blip_analysis import order_obj_instances
from blip_analysis import xy_coord_token, find_mask_tokens, generate_token_grid

COCO_PATH = "../../../datasets/coco"
BATCH_SIZE = 6
grid_size = 14

# TODO: 
# * get secondary object based on proximity - closest object
# * limit objects closer to the main cluster of activation
#   or choose the main object from area in main cluster of activation 
def get_bbox_center(annotation):
     return np.array((annotation["bbox"][0] + int(annotation["bbox"][2]/2), 
             annotation["bbox"][1] + int(annotation["bbox"][3]/2)))

def get_largest_mask(annotation, obj, img_draw):
    # select main (largest) mask for the object class        
    objects_area = [(ann['id'], ann['area']) for ann in annotation['annotations']['annotations'] if ann['category_id']==obj and ann['iscrowd']==0]
    largest_obj = sorted(objects_area, key=lambda x: x[1], reverse=True)[0]
    largest_obj_ann = [ann for ann in annotation['annotations']['annotations'] if ann['id']==largest_obj[0]]
    largest_obj_ann = largest_obj_ann[0] #avoid more than one object with same area
    segmentation = [ann['segmentation'] for ann in annotation['annotations']['annotations'] if ann['id']==largest_obj_ann["id"]]
    for segment in segmentation[0]:
        try:
            img_draw.polygon(segment, fill ="#ffffff")
        except ValueError:
            return None
    return [largest_obj_ann]
    
def get_all_masks(annotation, obj, img_draw):
    obj_annotation = [ann for ann in annotation['annotations']['annotations'] if ann['category_id']==obj]
    for ann in obj_annotation:
        segment = ann["segmentation"]
        if isinstance(segment, list):
            try:
                img_draw.polygon(segment[0], fill ="#ffffff")
            except ValueError:
                return None
    return obj_annotation

def get_mask_with_attention(img, annotation, category_id, img_draw, layer_attention_maps):
    objects_with_attention = order_obj_instances(img, image_annotations=annotation, category_id=category_id, layer_attention_map=layer_attention_maps)
    # list is ordered by max_attention, get the first element (object with max_attention)
    for segment in objects_with_attention[0][0]['segmentation']:
        img_draw.polygon(segment, fill ="#ffffff")
    #objects_with_attention[0=top attention](ann, att_score)
    return [objects_with_attention[0][0]]

def get_objects_mask(annotation, objects=[], option=1, attention_maps=None, layers=[11], split='train2017'):
    sample_image = Image.open(os.path.join(COCO_PATH, split, annotation['image']['file_name'])).convert('RGB')    
    masks = []
    obj_annotations = []

    try:
        for idx, obj in enumerate(objects):
            obj_annotation = []
            
            pil_mask = np.zeros(shape=sample_image.size, dtype=np.uint8)
            pil_mask = Image.fromarray(np.moveaxis(pil_mask, 0, -1))
            img_draw = ImageDraw.Draw(pil_mask) 
            
            if option == 1:
                obj_annotation = get_largest_mask(annotation, obj, img_draw)
                pil_mask = [pil_mask] #for compatibility
                

            # merge all masks of the object class into one
            elif option == 2:
                obj_annotation = get_all_masks(annotation, obj, img_draw)    
                pil_mask = [pil_mask] #for compatibility
                        
            # get closest mask based on a reference mask (first object)
            # calculate the centroid and then find the other mask closest to this centroid
            # using euclidian distance
            elif option == 3:
                ## assume first object is the reference. Get the first object based on biggest mask
                if idx==0:
                    obj_annotation = get_largest_mask(annotation, obj, img_draw)
                    pil_mask = [pil_mask] #for compatibility
                else:
                    # obj_annotation will be the idx 0 aka the main object
                    ref_center = get_bbox_center(obj_annotations[0][0])
                    distances = [(np.linalg.norm(ref_center - get_bbox_center(ann)), ann["id"]) for ann in annotation['annotations']['annotations'] if ann['category_id']==obj and ann['iscrowd']==0]
                    closest_obj = sorted(distances, key=lambda x: x[0], reverse=False)[0]
                    obj_annotation = [[ann for ann in annotation['annotations']['annotations'] if ann['id']==closest_obj[1] and ann['iscrowd']==0][0]]
                    for segment in obj_annotation[0]["segmentation"]:
                        try:
                            img_draw.polygon(segment, fill ="#ffffff")
                        except ValueError:
                            print(f"ValueError: {obj_annotation}")
                            return None, None
                    pil_mask = [pil_mask] #for compatibility
            
            elif option == 4:
                # this option produces 12 masks, one for each layer of the transformers encoder
                pil_mask = []
                for layer in layers:
                    att_layer_mask = np.zeros(shape=sample_image.size, dtype=np.uint8)
                    att_layer_mask = Image.fromarray(np.moveaxis(att_layer_mask, 0, -1))
                    img_draw = ImageDraw.Draw(att_layer_mask) 
                    obj_ann = get_mask_with_attention(sample_image, 
                                                      annotation, 
                                                      obj, 
                                                      img_draw, 
                                                      attention_maps[layer][0, 1:])
                    obj_annotation.append(obj_ann)
                    pil_mask.append(att_layer_mask)
            
            masks.append(pil_mask)
            obj_annotations.append(obj_annotation)
    except IndexError:
        print(f"Annotation : {annotation}")
        print(f"Object : {obj}")
        raise
        
    return masks, obj_annotations



# TODO: Improve logic of consistent tokens to account for different variations of layers and not only the last N
def find_tokens_in_region(attention_map:np.array, 
                          img: np.array, 
                          masks: list, #one mask per layer
                          layers=[9,10,11], 
                          min_n=3, 
                          max_n=20,
                          mask_threshold=.75,
                          grid_size=14,
                          patch_size=16,
                          n_tokens=196,
                          display_token_grid=False,
                          display_att_layers=False,
                          plot_axs=None,
                          plot_starting_row=0):
    if len(masks)!=len(layers):
        print("'mask' needs one mask per layer in 'layers'")
        return None, None, None
    
    tokens = {}
    all_top_n = []
    
    masks_tokens, masks_patches, imgs_patches = [], [], []
    for obj_mask in masks:
        # find which tokens belong to the mask
        mask_tokens_, mask_patches_, img_patches_ = find_mask_tokens(img, obj_mask, mask_threshold,n_tokens=n_tokens,grid_size=grid_size, patch_size=patch_size)        
        # If there are not tokens related to the mask, it is probably
        # because the mask region inside the patch is too small
        # try to find mask tokens again with threshold=0.
        #    e.g.: if one pixel falls into the mask patch it means it is a mask token/patch
        if not mask_tokens_:
            mask_tokens_, mask_patches_, img_patches_ = find_mask_tokens(img, obj_mask, mask_threshold=0,n_tokens=n_tokens,grid_size=grid_size, patch_size=patch_size)
            if not mask_tokens_:
                return None, None, None
        masks_tokens.append(mask_tokens_)
        masks_patches.append(mask_patches_)
        imgs_patches.append(img_patches_)
        
    # for each n_layers in the attention_map
    # find all tokens
    disp_idx = plot_starting_row
    mask_idx = 0
    for layer_no, mask_tokens, mask_patches, img_patches in zip(layers, masks_tokens, masks_patches, imgs_patches): 
        layer = attention_map[layer_no, :, :]
        tokens_layer_i = {}
        
        # MAX token whole image, excluding object area
        bg_att_map_mask = copy.deepcopy(layer[0, 1:])
        bg_att_map_mask[mask_patches] = 0
        tokens_layer_i['max_image'] = np.argmax(bg_att_map_mask)
        
        # set all background activation WITHIN background mask to -1
        img_att_map_mask = copy.deepcopy(layer[0, 1:])
        img_att_map_mask[~mask_patches] = 0
        
        # MIN MAX token within object/mask
        max_token_obj_layer_i = np.argmax(img_att_map_mask)
        # TODO: Improve logic to get min attention token considering only one region. 
        # when slicing np.array and using np.argmin the index returned is not relative to original array anymore
        # e.g.: np.argmin(img_map_fg[fg_mask_patch]) will return an index relative to the new array img_map_fg[fg_mask_patch]
        min_token_obj_layer_i = 0
        min_activation = np.max(img_att_map_mask)
        for token_i in mask_tokens:
            if img_att_map_mask[token_i] < min_activation:
                min_token_obj_layer_i = token_i
                min_activation = img_att_map_mask[token_i]

        tokens_layer_i['max_obj'] = max_token_obj_layer_i
        tokens_layer_i['min_obj'] = min_token_obj_layer_i
        
        # RANDOM token within object
        tokens_layer_i['random_obj'] = random.choice(mask_tokens)
        if len(mask_tokens)>=3:
            while tokens_layer_i['random_obj'] in [tokens_layer_i['max_obj'], tokens_layer_i['min_obj']]:
                tokens_layer_i['random_obj'] = random.choice(mask_tokens)
        
        # Get top N activations of the layer
        # based on https://stackoverflow.com/questions/6910641/how-do-i-get-indices-of-n-maximum-values-in-a-numpy-array
        # top_n = np.argpartition(img_att_map_mask, -1 * max_n)[-1 * max_n:]
        # top_n = top_n[np.argsort(img_att_map_mask[top_n])]
        # top_n = top_n[::-1]
        # all_top_n.append(top_n)  
        
        tokens[layer_no] = tokens_layer_i

        # show grid with all tokens found given the input mask
        if display_token_grid:
            token_grid = generate_token_grid(img_patches, list(mask_patches))
            
        if display_att_layers:
#             plot_axs[disp_idx, 0].imshow(masks[mask_idx])
            plot_axs[disp_idx, 1].imshow(masks[mask_idx])
            plot_axs[disp_idx, 1].axis('off')
            plot_axs[disp_idx, 2].imshow(token_grid)            
            plot_axs[disp_idx, 3].imshow(layer[0, 1:].reshape(grid_size, grid_size))
            plot_axs[disp_idx, 3].grid(visible=True)
            plot_axs[disp_idx, 4].imshow(img_att_map_mask.reshape(grid_size, grid_size))
            plot_axs[disp_idx, 4].set_title(f"layer {layer_no}, shape {img_att_map_mask.shape} ")
            plot_axs[disp_idx, 4].grid(visible=True)
            row_p, col_p = xy_coord_token(tokens_layer_i['min_obj'])
            plot_axs[disp_idx, 4].scatter(row_p, col_p, marker='_', s=400, c='white')
            row_p, col_p = xy_coord_token(tokens_layer_i['max_obj'])
            plot_axs[disp_idx, 4].scatter(row_p, col_p, marker='+', s=400, c='red')
            row_p, col_p = xy_coord_token(tokens_layer_i['random_obj'])
            plot_axs[disp_idx, 4].scatter(row_p, col_p, marker='*', s=400, c='green')
            row_p, col_p = xy_coord_token(tokens_layer_i['max_image'])
            plot_axs[disp_idx, 4].scatter(row_p, col_p, marker='P', s=400, c='orange')            
            disp_idx += 1
            mask_idx += 1            
    
    consistent_token = None
#     # FOR CONSISTENT TOKEN ACROSS SEVERAL LAYERS
#     # find if there's intersection with previous layer
#     for n in range(min_n, max_n, 2):
#         common_tokens = set()
#         for layer_no, layer_top_n in enumerate(all_top_n):
#             if layer_no==0:
#                 common_tokens = set(layer_top_n[:n])                
#             else:
#                 common_tokens = common_tokens.intersection(set(layer_top_n[:n]))
#                 if not common_tokens:
# #                     print(f"No common token with top {n}")
#                     break
#         if common_tokens:
# #             print(f"found common tokens with top {n}")
#             consistent_token = list(common_tokens)
#             if len(consistent_token)>1:
#                 # sum selected tokens across all layers to get top 1
#                 sums = dict(zip(consistent_token, [0] * len(consistent_token)))
#                 for layer in attention_map[n_layers:, :, :]:
#                     for token in consistent_token:
#                         sums[token] += layer[0, token].cpu().detach().numpy()
# #                 print(sums)
#                 consistent_token = max(sums, key=sums.get)
# #                 print(consistent_token)
#             else:
#                 consistent_token = consistent_token[0]
#             break    

    return tokens, consistent_token, all_top_n

def extract_tokens(img: Image.Image, 
                   object_mask:list, 
                   mean_att_map, 
                   layers:list,
                   grid_size=14,
                   patch_size=16, 
                   n_tokens=196,
                   mask_threshold=.75, 
                   debug=False, 
                   plot_axs=None,
                   plot_starting_row=0):
    # some regions of the image have intermediate values between 0 and 255
    # maybe resizing the image create these "intermediate" pixels.
    def separate_masks(mask):
        fg_mask_img = copy.deepcopy(mask)
        fg_mask_img[fg_mask_img==255] = 255
        fg_mask_img[fg_mask_img!=255] = 0
        fg_mask_img = fg_mask_img[:,:,np.newaxis]
        # get background mask by reversing the image mask
        bg_mask_img = copy.deepcopy(fg_mask_img)
        bg_mask_img[fg_mask_img==255] = 0
        bg_mask_img[fg_mask_img!=255] = 255
        return fg_mask_img, bg_mask_img
    
## Not using background tokens at the moment
#     bg_tokens, consistent_bg_token, _ = find_tokens_in_region(mean_att_map, np.array(img), bg_mask_img, mask_threshold=mask_threshold,
#                                                               display_token_grid=debug, display_att_layers=debug)
    fg_tokens, consistent_fg_token, _ = find_tokens_in_region(attention_map=mean_att_map, 
                                                              img=np.array(img), 
                                                              masks=[separate_masks(mask)[0] for mask in object_mask], 
                                                              layers=layers,
                                                              grid_size=grid_size,
                                                              patch_size=patch_size,
                                                              n_tokens=n_tokens,
                                                              mask_threshold=mask_threshold,
                                                              display_token_grid=debug, 
                                                              display_att_layers=debug,
                                                              plot_axs=plot_axs,
                                                              plot_starting_row=plot_starting_row)
    
    return fg_tokens, consistent_fg_token

     

    