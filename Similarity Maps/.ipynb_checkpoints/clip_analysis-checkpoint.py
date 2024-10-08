import requests
from transformers import CLIPProcessor, CLIPModel
from transformers import pipeline
import torch
import numpy as np
import copy
from PIL import Image, ImageDraw
import cv2
import os
import matplotlib.pyplot as plt

os.environ["WANDB_DISABLED"] = "true"

class ClipCaptioningModel():
    def __init__(self, pretrained_model="openai/clip-vit-base-patch16", output_encoder_attentions=True, output_encoder_hidden_states=True) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"model running on {self.device}")
        self.model, self.processor = self.model_initialization(pretrained_model, output_encoder_attentions, output_encoder_hidden_states)

    def model_initialization(self, pretrained_model, output_encoder_attentions=True, output_encoder_hidden_states=True):
        model = CLIPModel.from_pretrained(pretrained_model)
        model.config.output_attentions = output_encoder_attentions
        model.config.output_hidden_states = output_encoder_hidden_states
        processor = CLIPProcessor.from_pretrained(pretrained_model)
        
        model.to(self.device)

        ## only used for caption generation. not needed for regualr forward pass that gets the
        ## attention maps
        # max_length = 16
        # num_beams = 4
        # gen_kwargs = {"max_length": max_length, "num_beams": num_beams, "output_attentions": 'true'}

        return model, processor

    #requires more inputs
    def forward_pass(self, imgs:list):
        inputs = self.processor(text=[""], images=imgs, return_tensors="pt", padding=True)

        outputs = self.model(**inputs)

        return outputs   

    def get_all_attention_maps(self, attentions, renorm_weights=True):
        '''
        attentions: has to be ?
        '''
        # Average the attention weights across all heads.
        mean_att_map = torch.mean(attentions, dim=2)

        # To account for residual connections, we add an identity matrix to the
        # attention matrix and re-normalize the weights.        
        if renorm_weights:
            residual_att = torch.eye(mean_att_map.size(2))
            aug_mean_map = mean_att_map + residual_att.to(self.device)
            aug_mean_map = aug_mean_map / aug_mean_map.sum(dim=-1).unsqueeze(-1)
            mean_att_map = aug_mean_map

        return mean_att_map

    def get_joint_attention_map(self, attentions):
        '''
        attentions: has to be ?
        '''
        # preprocess attention maps
        aug_att_mat = self.get_all_attention_maps(attentions, renorm_weights=True)

        # Recursively multiply the weight matrices
        joint_attentions = torch.zeros(aug_att_mat.size())
        joint_attentions = joint_attentions.to(self.device)
        joint_attentions[0] = aug_att_mat[0]

        for n in range(1, aug_att_mat.size(0)):
            joint_attentions[n] = torch.matmul(aug_att_mat[n], joint_attentions[n-1])

        # Last layer attention map with all joint attentions
        return joint_attentions[-1]
    

def display_att_map(att_map, img_size:tuple, grid_size=14):
    # show CLS token against all other tokens except itself
    display_att_layer = att_map[0, 1:].reshape(grid_size, grid_size).cpu().detach().numpy()
    display_att_layer = cv2.resize(display_att_layer / display_att_layer.max(), img_size)

    return display_att_layer
    
def is_patch_within_mask(original_img_mask, patch_coord, mask_threshold=.70):
    patch_with_mask = original_img_mask[patch_coord[0]:patch_coord[1], patch_coord[2]:patch_coord[3],:]
    # previously was dividing by 768, why? 768 is the embbedding size...
    perc_mask_pixels = len(patch_with_mask[patch_with_mask==255]) / len(patch_with_mask.flatten())

    return perc_mask_pixels > mask_threshold, patch_with_mask

def find_original_img_patch(vit_patch:int, original_img:Image=None, grid_size:int=14, patch_size:int=16):
#     h_p, w_p = vit_patch
#     projected_patch = original_img[h_p * patch_size:(h_p * patch_size)+patch_size, w_p * patch_size:(w_p * patch_size)+patch_size]
    col_p = vit_patch // grid_size
    row_p = vit_patch - (col_p * grid_size)
    y = row_p * patch_size
    width = patch_size
    x = col_p * patch_size
    height = patch_size

    projected_patch = None
    if original_img is not None:
        projected_patch = original_img[x:x+width, y:y+height]

    return (x, x+width, y, y+height), projected_patch

def xy_coord_token(token, grid_size=14):
    y = token // grid_size
    x = token - (y * grid_size)
    return x, y

def find_mask_tokens(img, mask, mask_threshold, n_tokens = 196):
    """
    img and mask have to be resized to work (224,224)
    """
    img_patches = []
    mask_patches = np.zeros((n_tokens), dtype="bool")
    mask_tokens = []
    for patch_i in range(n_tokens):
        coord, img_patch = find_original_img_patch(vit_patch=patch_i, original_img=img)
        mask_patches[patch_i] = is_patch_within_mask(mask, coord, mask_threshold)[0]
        if mask_patches[patch_i]:
            mask_tokens.append(patch_i)
        img_patches.append(img_patch)
        
    return mask_tokens, mask_patches, img_patches    

def create_fg_mask(img_size, annotation) -> Image:
    pil_mask = np.zeros(shape=img_size, dtype=np.uint8)
    pil_mask = Image.fromarray(np.moveaxis(pil_mask, 0, -1))
    img_draw = ImageDraw.Draw(pil_mask)     
    for segm in annotation['segmentation']:
        if isinstance(segm, list):
            img_draw.polygon(segm, fill ="#ffffff")
    
    return pil_mask

def pre_process_mask(mask: Image, new_size=(224,224))-> np.ndarray:    
    fg_mask_img = copy.deepcopy(np.array(mask.resize(new_size)))
    # make it binary 255/0
    fg_mask_img[fg_mask_img!=255] = 0
    fg_mask_img = fg_mask_img[:,:,np.newaxis]
    
    return fg_mask_img

# orders object instances by attention.
# TODO: improve function name, not very descriptive 
def order_obj_instances(img: Image, image_annotations, layer_attention_map, category_id:int=-1):
    obj_instance_attention = {}
    if category_id>0:
        object_annotations = [ann for ann in image_annotations['annotations']['annotations'] if ann['iscrowd']==0 and ann['category_id']==category_id]
    else:
        object_annotations = [ann for ann in image_annotations['annotations']['annotations'] if ann['iscrowd']==0]

    for idx, ann in enumerate(object_annotations):
        mask = create_fg_mask(img.size, ann)
        mask = pre_process_mask(mask)
        mask_patches = find_mask_tokens(np.array(img.resize((224,224))), mask, .0)[1]
        
        obj_att_map = copy.deepcopy(layer_attention_map)
        # put into [0, 1] scale
        obj_att_map = obj_att_map / obj_att_map.max()
        # mask background
        obj_att_map[~mask_patches] = 0.0
        max_attention = np.max(obj_att_map)        
        obj_instance_attention[idx] = max_attention
    
    obj_instance_sorted = sorted(obj_instance_attention.items(), key=lambda x:x[1], reverse=True)
    obj_instance_sorted = [(object_annotations[instance[0]],instance[1]) for instance in obj_instance_sorted]
    return obj_instance_sorted       

def generate_token_grid(img_patches, mask_patches=[]):
    #if no mask patches plot the original image without masking anything    
    if not mask_patches:
        mask_patches = [True] * 196
        img_patch_color = "black"
        img_patch_border = 0
    else:
        img_patch_color = "red"
        img_patch_border = 1.5
    black_patch = np.zeros_like(img_patches[0])
    fig, axs = plt.subplots(nrows=14, ncols=14, figsize=(4, 4))
    for patch_i, img_patch in enumerate(img_patches):
        row_p, col_p = xy_coord_token(patch_i)
        if mask_patches[patch_i]:
            axs[col_p, row_p].imshow(img_patch)    
            axs[col_p, row_p].spines['top'].set_linewidth(img_patch_border)
            axs[col_p, row_p].spines['top'].set_color(img_patch_color)
            axs[col_p, row_p].spines['right'].set_linewidth(img_patch_border)
            axs[col_p, row_p].spines['right'].set_color(img_patch_color)
            axs[col_p, row_p].spines['bottom'].set_linewidth(img_patch_border)
            axs[col_p, row_p].spines['bottom'].set_color(img_patch_color)
            axs[col_p, row_p].spines['left'].set_linewidth(img_patch_border)
            axs[col_p, row_p].spines['left'].set_color(img_patch_color)                    
        else:
            axs[col_p, row_p].imshow(black_patch)
#             axs[col_p, row_p].axis('off')
        # Hide X and Y axes label marks
        axs[col_p, row_p].xaxis.set_tick_params(labelbottom=False)
        axs[col_p, row_p].yaxis.set_tick_params(labelleft=False)
        # Hide X and Y axes tick marks
        axs[col_p, row_p].set_xticks([])
        axs[col_p, row_p].set_yticks([])
    img_buf = io.BytesIO()
    plt.savefig(img_buf, bbox_inches='tight', format='png')
    plt.close(fig)
    
    return Image.open(img_buf)