#!/usr/bin/env python
# coding: utf-8


from torch import nn, Tensor
from PIL import Image
from pycocotools.coco import COCO
import numpy as np
import pickle
import torch
from matplotlib import pyplot as plt
from transformers import AutoProcessor, BlipForConditionalGeneration
from utils_BLIP_L import *

setting = 'train'

if(setting == 'train'):
    coco_annotation_file_path = "../../coco/images/annotations/instances_train2017.json"
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)
    img_base = '../../coco/images/train2017/'
    ids = coco_annotation.getImgIds()
    annotations = coco_annotation.loadImgs(ids[:40000])
    output_dir = 'Features/BLIP_Large/Train'
    
elif(setting == 'val'):
    coco_annotation_file_path = "../../coco/images/annotations/instances_val2017.json"
    coco_annotation = COCO(annotation_file=coco_annotation_file_path)
    img_base = '../../coco/images/val2017/'
    ids = coco_annotation.getImgIds()
    annotations = coco_annotation.loadImgs(ids)
    output_dir = 'Features/BLIP_Large/Val'

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
layers = np.arange(0,24)
feature_extractor = FeatureExtractor(model,layers)

generate_features(model,processor,feature_extractor, coco_annotation,annotations,img_base,output_dir)


