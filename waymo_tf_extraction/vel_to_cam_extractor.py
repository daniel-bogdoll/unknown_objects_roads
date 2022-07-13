import numpy as np
import math
import cv2
import io
import sys

from simple_waymo_open_dataset_reader import WaymoDataFileReader
from simple_waymo_open_dataset_reader import dataset_pb2, label_pb2
from simple_waymo_open_dataset_reader import utils
import matplotlib.cm

import pickle
from simple_waymo_open_dataset_reader import dataset_pb2 as open_dataset
import glob, argparse, tqdm, pickle, os 
from multiprocessing import Pool 

fnames = None 
CALIBRATION_PATH = None 


def convert(idx):
    global fnames
    fname = fnames[idx]
    datafile = WaymoDataFileReader(fname)
    for frame_id,frame in enumerate(datafile):
        # Get the top laser information
        laser_name = dataset_pb2.LaserName.TOP
        laser = utils.get(frame.lasers, laser_name)
        laser_calibration = utils.get(frame.context.laser_calibrations, laser_name)

        # Parse the top laser range image and get the associated projection.
        ri, camera_projection, range_image_pose = utils.parse_range_image_and_camera_projection(laser)

        # Convert the range image to a point cloud.
        pcl, pcl_attr = utils.project_to_pointcloud(frame, ri, camera_projection, range_image_pose, laser_calibration)
        
        
        # Get the front camera information
        camera_name = dataset_pb2.CameraName.FRONT
        camera_calibration = utils.get(frame.context.camera_calibrations, camera_name)
        camera = utils.get(frame.images, camera_name)

        # Get the transformation matrix for the camera.
        vehicle_to_image = utils.get_image_transform(camera_calibration)
        with open(os.path.join(CALIBRATION_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)), 'wb') as f:
            pickle.dump(vehicle_to_image, f)

def main(args):
    global fnames 
    fnames = list(glob.glob(args.record_path + "/*.tfrecord"))
    #fnames = list(glob.glob(args.record_path))
    print("Number of files {}".format(len(fnames)))
    

    with Pool(128) as p: # change according to your cpu
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--record_path', type=str, required=True)

    args = parser.parse_args()
    

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    
    CALIBRATION_PATH = os.path.join(args.root_path, "calibration")

    if not os.path.isdir(CALIBRATION_PATH):
        os.mkdir(CALIBRATION_PATH)
    
    main(args)