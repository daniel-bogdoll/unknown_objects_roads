from tqdm import tqdm
import pickle
import os
import numpy as np
from typing import Optional
import typer

def main(directory_path: str = typer.Option(...)):
    
    directory = os.listdir(directory_path)

    anomalies = []
    anomalies_count = 0

    for file_name in tqdm(directory):
        
        bbox_file_path = directory_path + file_name + "/" + file_name + ".pkl"
    
        with open(bbox_file_path, 'rb') as b:
            bbox_calibration = pickle.load(b)
            
            anomalies_count += np.count_nonzero(bbox_calibration['cluster_bboxes'][:, -1] == -1) 

            if -1 in bbox_calibration['cluster_bboxes'][:, -1]:
                anomalies.append(file_name)
            
    with open(directory_path + 'anomalies' + '.pkl', 'wb') as b:
            pickle.dump(anomalies, b, protocol=pickle.HIGHEST_PROTOCOL)

    with open(directory_path  + 'count_anomalies' + '.pkl', 'wb') as b:
            pickle.dump(anomalies_count, b, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    typer.run(main)