import pandas as pd
import os
import pickle
from PIL import Image, ImageDraw
import numpy as np
import math
import requests
from transformers import CLIPProcessor, CLIPModel
import os
import pickle
import numpy as np
import re
from tqdm import tqdm
import torch
from typing import Optional
import typer


def main(directory_path: str = typer.Option(...)):

    device = "cuda" if torch.cuda.is_available() else "cpu"
            
    directory = os.listdir(directory_path)

    #Load pre-trained CLIP model
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    #Found waymo labels
    labels = ["car", "traffic light", "person", "truck", "bus", "fire hydrant", "bicycle", "handbag", "backpack", "parking meter", "stop sign", "umbrella", "motorcycle", "tree", "bush"]

    for folder in tqdm(directory):
    
        object_path = directory_path + "/" + folder + "/objects/"
    
        bbox_file_path = directory_path + "/" + folder +'/{}.pkl'.format(folder)
    
        with open(bbox_file_path, 'rb') as b:
            bbox_calibration = pickle.load(b)
        
        
        object_dict = {}
    
        object_directory = os.listdir(object_path)

        #Give each area of the image with an object that was not identified in 3D to the CLIP model
        for object_file in sorted(object_directory):
        
            image = Image.open(object_path + str(object_file))
            
            #define prompt
            inputs = processor(text=[f"a photo of a {x} on a street" for x in labels], images=image, return_tensors="pt", padding=True)
            inputs.to(device)
            
            outputs = model(**inputs)
            
            logits_per_image = outputs.logits_per_image 
            probs = logits_per_image.softmax(dim=1).detach().cpu().numpy() 
            
            predicted_idx = int(np.argmax(probs, axis=1))
            
            predicted_probs = np.max(probs, axis=1)
            
            if predicted_probs[0] < 0.25:
                predicted_label = "unknown object"
                predicted_label_old = labels[predicted_idx]
                predicted_probs_old = predicted_probs
                predicted_probs = 1 - predicted_probs
                prediction_flag = -1
                
                search_nr_of_bbox = re.search(r"\d+(\.\d+)?", object_file)
                nr_off_bbox = int(search_nr_of_bbox.group(0))
                bbox_calibration['cluster_bboxes'][nr_off_bbox][-1] = prediction_flag
                
                object = {
                "id": nr_off_bbox,
                "prediction": predicted_label,
                "prediction probability": predicted_probs[0],
                "prediction flag": prediction_flag,
                "box": bbox_calibration['cluster_bboxes'][nr_off_bbox],  
                "prediction old": predicted_label_old,
                "prediction probability old": predicted_probs_old[0],
                }
                
            else:
                predicted_label = labels[predicted_idx]
                predicted_probs = predicted_probs
                prediction_flag = 1
                
                search_nr_of_bbox = re.search(r"\d+(\.\d+)?", object_file)
                nr_off_bbox = int(search_nr_of_bbox.group(0))
                bbox_calibration['cluster_bboxes'][nr_off_bbox][-1] = prediction_flag
                
                object = {
                "id": nr_off_bbox,
                "prediction": predicted_label,
                "prediction probability": predicted_probs[0],
                "prediction flag": prediction_flag,
                "box": bbox_calibration['cluster_bboxes'][nr_off_bbox]   
                }       
            
            object_dict["object_" + str(nr_off_bbox)] = object
            
        bbox_calibration['objects'] = object_dict
    
        with open(bbox_file_path, 'wb') as b:
            pickle.dump(bbox_calibration, b, protocol=pickle.HIGHEST_PROTOCOL)
    
if __name__ == "__main__":
    typer.run(main)