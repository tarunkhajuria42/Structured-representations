**Readme **

To replicate the experiments used in the paper please follow the following instructions.

1. Setting up environment
-> install the environment in anaconda using the requirements.txt file

2. Dataset and Metadata
-> We use the train set from the COCO 2017 dataset (given here: https://cocodataset.org/#download). 
-> The meta-data for the tasks can be obtained by running the script 'get_image_names.py'; please setup the input and output folders in the script.

3. Extract tokens and relevant features
-> Use the scripts features_extraction_****model****.py with variations BLIP, CLIP and CNN to extract the tokens from different pre-trained models for the tasks.

4. Train and evaluate probe
- Use the files probe_***model experiment***.py with variations BLIP,CLIP,CNN and BLIP_captions_experiments to run the probing experiments on the extracted representations.

5. Plot the results
- Use the notebooks Results_**model** with variations BLIP,CLIP,CNN and ALL to obtain the final plots with the results. The results plots in the main paper can be plotted using notebooks BLIP and ALL. Supplementary plots using CLIP and CNN notebooks. Obtained Results folders are already included for plotting to be possible.


