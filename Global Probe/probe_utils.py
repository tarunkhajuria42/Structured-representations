from sklearn import *
import os
import pickle
from sklearn.utils import shuffle
from sklearn.linear_model import Perceptron
from nltk.stem import WordNetLemmatizer
import nltk
from pycocotools.coco import COCO
nltk.download('wordnet')

cat_id_to_name = {
  1: "person",
  2: "bicycle",
  3: "car",
  4: "motorcycle",
  5: "airplane",
  6: "bus",
  7: "train",
  8: "truck",
  9: "boat",
  10: "traffic light",
  11: "fire hydrant",
  13: "stop sign",
  14: "parking meter",
  15: "bench",
  16: "bird",
  17: "cat",
  18: "dog",
  19: "horse",
  20: "sheep",
  21: "cow",
  22: "elephant",
  23: "bear",
  24: "zebra",
  25: "giraffe",
  27: "backpack",
  28: "umbrella",
  31: "handbag",
  32: "tie",
  33: "suitcase",
  34: "frisbee",
  35: "skis",
  36: "snowboard",
  37: "sports ball",
  38: "kite",
  39: "baseball bat",
  40: "baseball glove",
  41: "skateboard",
  42: "surfboard",
  43: "tennis racket",
  44: "bottle",
  46: "wine glass",
  47: "cup",
  48: "fork",
  49: "knife",
  50: "spoon",
  51: "bowl",
  52: "banana",
  53: "apple",
  54: "sandwich",
  55: "orange",
  56: "broccoli",
  57: "carrot",
  58: "hot dog",
  59: "pizza",
  60: "donut",
  61: "cake",
  62: "chair",
  63: "couch",
  64: "potted plant",
  65: "bed",
  67: "dining table",
  70: "toilet",
  72: "TV",
  73: "laptop",
  74: "mouse",
  75: "remote",
  76: "keyboard",
  77: "cell phone",
  78: "microwave",
  79: "oven",
  80: "toaster",
  81: "sink",
  82: "refrigerator",
  84: "book",
  85: "clock",
  86: "vase",
  87: "scissors",
  88: "teddy bear",
  89: "hair drier",
  90: "toothbrush"
}


def get_caption_for_id(imgid,coco_captions):
    capIds = coco_captions.getAnnIds(imgIds=imgid)
    captions = coco_captions.loadAnns(capIds)
    # Extract captions
    caps_list = [ann['caption'] for ann in captions]
    caps = ' '.join(caps_list) 
    return caps

def word_in_caption(word,caption,wnl):
    caption = caption.lower().split()
    word = wnl.lemmatize(word.lower())
    caption = list(map(wnl.lemmatize, caption))
    return (word in caption)

def get_features_for_layer(input_dir, layer, sel_set,val=False):
    wnl = WordNetLemmatizer()
    coco_captions  = COCO( "../../coco/images/annotations/captions_val2017.json")
    files = os.listdir(input_dir)
    holder = {}
    # open each file in set
    holder['label']=[]
    holder['nc_label']=[]
    for ind,file in enumerate(files):
        try:
            with open(input_dir+file,'rb') as f:
                data = pickle.load(f)
        except:
            continue
            # iterate over images in batch
        for image in data:
            img_id = image['id']
            caption = get_caption_for_id(img_id,coco_captions)
            # iterate over the objects in the image
            try:
                new = image[layer]
            except:
                continue
            for cat in image[layer].keys():
                # add label to the labels array
                if(cat in ['cls_features','all_random']):
                    continue
                if(cat not in sel_set):
                    continue
                # add vector to the
                if('random_features' not in holder.keys()):
                    holder['random_features'] = []
                    holder['all_features'] = []
                    if(val):
                        holder['nc_random_features'] = []
                        holder['nc_all_features'] = []
                if('random_features' not in image[layer][cat].keys()):
                    continue
                if(val):
                    if(not word_in_caption(cat_id_to_name[cat],caption,wnl)):
                        holder['nc_label'].append(cat)
                        holder['nc_random_features'].append(image[layer][cat]['random_features'])
                        holder['nc_all_features'].append(image[layer][cat]['all_features'])
                        continue

                holder['label'].append(cat)
                holder['random_features'].append(image[layer][cat]['random_features'])
                holder['all_features'].append(image[layer][cat]['all_features'])

    return holder

def model_results(X_train,y_train,X_val,y_val,X_val_nc,y_val_nc,caption = False,final=False,f1 = False):
    clf = Perceptron(tol=1e-3, random_state=0)
    clf.fit(X_train, y_train)
    if(final):
        if(f1):
            preds = clf.predict(X_test)
            return f1_score(y_test,preds)
        else:    
            return clf.score(X_test, y_test)
    else:
        if(f1):
            preds = clf.predict(X_val)
            return f1_score(y_val,preds)
        elif(caption):
            return clf.score(X_val, y_val),clf.score(X_val_nc, y_val_nc)
        else:
            return clf.score(X_val, y_val)
            
