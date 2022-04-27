# Written by Enrico Eisen based on Grégoire Payen de La Garanderie, Durham University
#
# Licensed under the Apache License, Version 2.0;
# ==============================================================================
import os
from typing import Optional

from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import math
import numpy as np
import pandas as pd
import pickle
import torch
import typer


def display_labels_on_image(vehicle_to_image, object_path, img, img_unproc, boxes):
    """
    Calls for each bounding box the desired function that drwas the box (2D or 3D)
    """
    # Draw all the groundtruth labels
    for box_nr in range(boxes.shape[0]):
        draw_3d_box(img, img_unproc, box_nr, vehicle_to_image, object_path, boxes[box_nr], draw_2d_bounding_box=True)

def draw_3d_box(img, img_unproc, box_nr, vehicle_to_image, object_path,  box, colour=(0,0,255), draw_2d_bounding_box=False):
    """Draw a 3D bounding from a given 3D label on a given "img". "vehicle_to_image" must be a projection matrix from the vehicle reference frame to the image space.

    draw_2d_bounding_box: If set a 2D bounding box encompassing the 3D box will be drawn
    """

    vertices = get_3d_box_projected_corners(vehicle_to_image, box)
    if vertices is None:
        # The box is not visible in this image
        return
    
    if draw_2d_bounding_box:
        x1,y1,x2,y2 = compute_2d_bounding_box(img.shape, vertices)
        
        # Coordinates to cut out resulting rectangle areas from bounding boxes but enlarge area to fully caputure the object
        y1_cut = int(np.floor(y1-y1*0.01))
        y2_cut = int(np.ceil(y2+y2*0.01))
        x1_cut = int(np.floor(x1-x1*0.01))
        x2_cut = int(np.ceil(x2+x2*0.01))
        
        # Cut and Save rectangle area 
        img_unproc.crop((x1_cut, y1_cut, x2_cut, y2_cut)).save(object_path + "object_" + str(box_nr) + ".jpg")
        

        if (x1 != x2 and y1 != y2):
            # Draw the 2D bounding box rectangle 
            cv2.rectangle(img, (x1,y1), (x2,y2), color=(0,0,255), thickness = 1)
    else:
        # Draw the edges of the 3D bounding box
        for k in [0, 1]:
            for l in [0, 1]:
                for idx1,idx2 in [((0,k,l),(1,k,l)), ((k,0,l),(k,1,l)), ((k,l,0),(k,l,1))]:
                    cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
        # Draw a cross on the front face to identify front & back.
        for idx1,idx2 in [((1,0,0),(1,1,1)), ((1,1,0),(1,0,1))]:
            cv2.line(img, tuple(vertices[idx1]), tuple(vertices[idx2]), colour, thickness=1)
    
def get_3d_box_projected_corners(vehicle_to_image, box):
    """Get the 2D coordinates of the 8 corners of a label's 3D bounding box.

    vehicle_to_image: Transformation matrix from the vehicle frame to the image frame.
    label: The object label
    """

    # Get the vehicle pose
    box_to_vehicle = get_box_transformation_matrix(box)

    # Calculate the projection from the box space to the image space.
    box_to_image = np.matmul(vehicle_to_image, box_to_vehicle)


    # Loop through the 8 corners constituting the 3D box
    # and project them onto the image
    vertices = np.empty([2,2,2,2])
    for k in [0, 1]:
        for l in [0, 1]:
            for m in [0, 1]:
                # 3D point in the box space
                v = np.array([(k-0.5), (l-0.5), (m-0.5), 1.])

                # Project the point onto the image
                v = np.matmul(box_to_image, v)

                # If any of the corner is behind the camera, ignore this object.
                if v[2] < 0:
                    return None

                vertices[k,l,m,:] = [v[0]/v[2], v[1]/v[2]]

    vertices = vertices.astype(np.int32)

    return vertices

def compute_2d_bounding_box(img_or_shape,points):
    """Compute the 2D bounding box for a set of 2D points.
    
    img_or_shape: Either an image or the shape of an image.
                  img_or_shape is used to clamp the bounding box coordinates.
    
    points: The set of 2D points to use
    """

    if isinstance(img_or_shape,tuple):
        shape = img_or_shape
    else:
        shape = img_or_shape.shape

    # Compute the 2D bounding box and draw a rectangle
    x1 = np.amin(points[...,0])
    x2 = np.amax(points[...,0])
    y1 = np.amin(points[...,1])
    y2 = np.amax(points[...,1])

    x1 = min(max(0,x1),shape[1])
    x2 = min(max(0,x2),shape[1])
    y1 = min(max(0,y1),shape[0])
    y2 = min(max(0,y2),shape[0])

    return (x1,y1,x2,y2)

def get_box_transformation_matrix(box):
    """Create a transformation matrix for a given label box pose."""

    tx,ty,tz = box[0],box[1],box[2]
    
    sl, sh, sw = box[3], box[5], box[4]   

    c = math.cos(0)
    s = math.sin(0)
  
    return np.array([
        [ sl*c,-sw*s,  0,tx],
        [ sl*s, sw*c,  0,ty],
        [    0,    0, sh,tz],
        [    0,    0,  0, 1]])
    
    """ return np.array([
        [ sl,   -sw,    0,    tx],
        [ sl,    sw,    0,    ty],
        [  0,     0,   sh,    tz],
        [  0,     0,    0,    1]]) """


def main(directory_path: str = typer.Option(...), camera_path: str = typer.Option(...), calibration_path: str = typer.Option(...), clustering_path: str = typer.Option(...)):

    os.mkdir(directory_path) #target directory

    directory = os.listdir(camera_path) #directory with images

    for image_file in tqdm(directory):
    
        image_file = image_file.split('.')
        filename = image_file[0]
        
        image_path = directory_path + filename + "/"
        os.mkdir(image_path)
        
        object_path = image_path + "objects/"
        os.mkdir(object_path)
        
        camera_file_path = camera_path+'/{}.jpg'.format(filename)
        image = cv2.imread(camera_file_path)
        
        image_unproc = Image.open(camera_file_path)
        
        bbox_file_path = clustering_path+'/{}.pkl'.format(filename)
        
        with open(bbox_file_path, 'rb') as b:
            bbox_calibration = pickle.load(b)
            
        calibration_file_path = calibration_path+'/{}.pkl'.format(filename)
        with open(calibration_file_path, 'rb') as b:
            vehicle_to_image = pickle.load(b)
        
         # Display the 2D bounding box on the image.
        display_labels_on_image(vehicle_to_image, object_path, image, image_unproc, bbox_calibration['cluster_bboxes'])
        cv2.imwrite(image_path + filename + ".jpg", image)
        
        # Add new collom at the end
        bbox_calibration['cluster_bboxes'] = np.c_[ bbox_calibration['cluster_bboxes'], np.zeros(bbox_calibration['cluster_bboxes'].shape[0])]
        
        with open(image_path + filename + '.pkl', 'wb') as b:
            pickle.dump(bbox_calibration, b, protocol=pickle.HIGHEST_PROTOCOL)
    
    
if __name__ == "__main__":
    typer.run(main)
    

