# Author: Christin Scheib


import os
import pickle
from PIL import Image, ImageDraw
import cv2
import matplotlib.cm
import numpy as np

cmap = matplotlib.cm.get_cmap("viridis")

# pcl to image projection
def display_laser_on_image(img, pcl, vehicle_to_image):
    # Convert the pointcloud to homogeneous coordinates.
    pcl1 = np.concatenate((pcl,np.ones_like(pcl[:,0:1])),axis=1)

    # Transform the point cloud to image space.
    proj_pcl_all = np.einsum('ij,bj->bi', vehicle_to_image, pcl1) 
    

    # Project the point cloud onto the image.
    proj_pcl_all = proj_pcl_all[:,:2]/proj_pcl_all[:,2:3]

    return proj_pcl_all


road_coordinates_path = "/<path>/road_coordinates/waymo/testing2"
camera_path = "/<path>/waymo_extraction_all/testing/camera"
lidar_path = "/<path>/waymo_extraction_all/testing/lidar"
calibration_path = "/<path>/waymo_extraction_calibration/testing/calibration"

directory_road = os.listdir(road_coordinates_path)
print(len(directory_road))
progress = 0

for file in directory_road:
    # Show progress
    progress += 1 
    print("Processing image ", progress, " /", len(directory_road))

    # Open the pkl that contains the road coordinates from the segmentation
    road_file_path = road_coordinates_path + '/' + file
    file = file.split('.')
    filename = file[0]
    
    print(filename)
    with open(road_file_path, 'rb') as a:
        road_coordinates = pickle.load(a)
    
    calibration_file_path = calibration_path+'/{}.pkl'.format(filename)
    camera_file_path = camera_path+'/{}.jpg'.format(filename)
    lidar_file_path = lidar_path+'/{}.pkl'.format(filename)

    try:
        # Open calibration data
        with open(calibration_file_path, 'rb') as b:
            vehicle_to_image = pickle.load(b)
        
        # Open the image as cv2 and Pillow file
        image = cv2.imread(camera_file_path)
        img = Image.open(camera_file_path)

        # Open lidar data
        with open(lidar_file_path, 'rb') as c:
            pcl = pickle.load(c)
            pcl = pcl["lidars"]['points_xyz']


        #load a pixel mask of the image
        pixels = img.load()

        for i in range(len(road_coordinates)):
            list_of_coords = []
            for j in range(len(road_coordinates[i])):
                list_of_coords.append(tuple(road_coordinates[i][j]))
            draw = ImageDraw.Draw(img)
            points = tuple(list_of_coords)
            polygon = draw.polygon((points), fill=200)
        
        #convert each point from the complete point cloud to a tuple
        point_tuple = tuple([tuple(point) for point in pcl])
    
        # Convert the point cloud to image coordinates
        pts2d = display_laser_on_image(image, pcl, vehicle_to_image)
        #convert the converted points to a list instead of array
        pts2d=pts2d.tolist()
        
       

        #create a dictionary with the original point from the pcl as key and the projected point as value
        matching_points = {}
        for i in range(len(point_tuple)):
            matching_points[point_tuple[i]]=pts2d[i]

        # delete points that are outside the image range
        for k,v in list(matching_points.items()):
            if v[0] >= img.width: 
                del matching_points[k]
            elif v[0] < 0:
                del matching_points[k]
            elif v[1] >= img.height:
                del matching_points[k]
            elif v[1] < 0:
                del matching_points[k]
        
        #each point which is on the street, which is colorcoded, is saved in an empty list
        road_list_3d = []
        for key, value in matching_points.items():
            
            x = value[0]
            y = value [1]
            if pixels[x,y] == (200, 0, 0):
                street_point_3d = key
                road_list_3d.append(street_point_3d)
      
        # save all the points from the point cloud that are on the street
        with open('/<path>/waymo_extraction_all/testing/results/final_road_mask/{}.pkl'.format(filename), 'wb') as f:
            pickle.dump(road_list_3d, f)
       
      

     
   

    except:
        print('file not found')

    
      

    
