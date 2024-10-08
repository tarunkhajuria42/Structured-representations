# README

## Requirements
* Unzip the COCO dataset (annotations and images) to a local folder.
* Set the global variable COCO_PATH in the file `generate_token_similarity.py` pointing to the local folder with COCO data.

## How to generate similarity maps
run the following python script to generate the similarity maps for all models:

```
python generate_token_similarity.py
```