from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse, time, pickle, os 
import numpy as np
import open3d as o3d
import copy

import sys 
from os import path
sys.path.append(path.relpath('./'))
sys.path.append(path.relpath('./CenterPoint'))
from CenterPoint.tools.visual import center_to_corner_box3d, label2color, corners_to_lines

import matplotlib.pyplot as plt

CLUSTER_PATH = None 
LIDAR_PATH = None
VISUALIZATION_PATH = None

def front_view(pcd):
    """
    Crops the point cloud to a front view

    Args:
        pcd (open3d.geometry.PointCloud):  LiDAR point cloud 

    Returns:
        open3d.geometry.PointCloud: Point cloud in front of vehicle (i.e. x-coords ≥ 0)
    """
    max_bounds = pcd.get_max_bound()
    min_bounds = pcd.get_min_bound()
    front_bb = o3d.geometry.AxisAlignedBoundingBox(np.array([0, min_bounds[1], min_bounds[2]]), max_bounds)
    cropped_pcd = pcd.crop(front_bb)
    return cropped_pcd

def estimate_ground(pcd, distance_threshold=0.5, ransac_n=10, num_iterations=500):
    """
    Estimates the ground of a point cloud via RANSAC algorithm

    Args:
        pcd (_type_): LiDAR point cloud 
        distance_threshold (float, optional): Defines the maximum distance a point can have to an estimated plane to be considered an inlier. Defaults to 0.5.
        ransac_n (int, optional): Defines the number of points that are randomly sampled to estimate a plane. Defaults to 10.
        num_iterations (int, optional): Defines how often a random plane is sampled and verified. Defaults to 500.

    Returns:
        tuple(open3d.geometry.PointCloud, open3d.geometry.PointCloud): Point cloud of estimated ground and point cloud not considered as ground
    """
    ground_model, ground_mask = pcd.segment_plane(distance_threshold=distance_threshold,
                                         ransac_n=ransac_n,
                                         num_iterations=num_iterations)
    pcd_ground = pcd.select_by_index(ground_mask)
    pcd_without_ground = pcd.select_by_index(ground_mask, invert=True)
    return pcd_ground, pcd_without_ground

def remove_statistical_outlier(pcd, nb_neighbors=25, std=8):
    """
    Removes statistical outliers in point cloud

    Args:
        pcd (open3d.geometry.PointCloud): _description_
        nb_neighbors (int, optional): Specifies how many neighbors are taken into account in order to calculate the average distance for a given point. Defaults to 25.
        std (int, optional): _description_. Defaults to 8.

    Returns:
        open3d.geometry.PointCloud: Point cloud without outliers
    """
    cl, ind = pcd.remove_statistical_outlier(nb_neighbors,std)
    return pcd.select_by_index(ind)

def distance_filter(pcd, filter_pcd, offset=0, invert=False):
    """
    Filters point cloud by another point cloud and a given offset between the two clouds.

    Args:
        pcd (open3d.geometry.PointCloud): Point cloud to filter
        filter_pcd (open3d.geometry.PointCloud): Filter point cloud
        offset (int, optional): The distance threshold between points. Defaults to 0.
        invert (bool, optional): Return the points with distance greater than offset. Defaults to False.

    Returns:
        open3d.geometry.PointCloud: Filtered point cloud
    """
    distances = np.asarray(pcd.compute_point_cloud_distance(filter_pcd))
    index = np.argwhere(distances <= offset)
    return pcd.select_by_index(index, invert=invert)

def flatten_pcd(pcd):
    """
    Remove the tird dimension (i.e. sets the z-axis to zero) and thus flattens the points cloud. 
    
    This does not mean the resulting point cloud is a 2-dimensional vector but a 3-dimensional
    with the last axis full of zeros

    Args:
        pcd (open3d.geometry.PointCloud): Point cloud to flatten

    Returns:
        open3d.geometry.PointCloud: Flattend point cloud
    """
    flattened_pcd = copy.deepcopy(pcd)
    points = np.asarray(flattened_pcd.points)
    points[:,2] = 0
    flattened_pcd.points = o3d.utility.Vector3dVector(points)
    return flattened_pcd

def alpha_surface_reconstruction(pcd, alpha=10):
    """
    Estimates the concave hull of the a flattened point cloud by an alpha shape estimation 
    [Edelsbrunner1983]. Used for estimating the road surface. 

    Args:
        pcd (open3d.geometry.PointCloud): Flattened point cloud to base the estimation on
        alpha (int, optional): Tradeoff parameter for surface granularity. Defaults to 10.

    Returns:
        (o3d.geometry.TriangleMesh, o3d.geometry.TriangleMesh): Flattened and unflattened alpha shape surface
    """
    road_mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    vertices = np.asarray(copy.deepcopy(road_mesh.vertices)) #flatten road mesh
    vertices[:,2] = 0
    flattend_mesh = copy.deepcopy(road_mesh)
    flattend_mesh.vertices = o3d.utility.Vector3dVector(vertices)
    return flattend_mesh, road_mesh

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo 3D clustering')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument("--road_path", type=str, required=True)
    parser.add_argument('--output_path', type=str, required=False)
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    if not os.path.isdir(args.root_path):
        os.mkdir(args.root_path)

    LIDAR_PATH = os.path.join(args.root_path, 'lidar')
    if args.output_path:
        ROAD_MESH_PATH = args.output_path
    else:
        ROAD_MESH_PATH = os.path.join(args.root_path, 'road_mesh')

    assert os.path.isdir(LIDAR_PATH)
    assert os.path.isdir(args.road_path)
    
    if not os.path.isdir(ROAD_MESH_PATH):
            os.mkdir(ROAD_MESH_PATH)

    counter = 0
    frame_names = [os.path.splitext(filename)[0] for filename in os.listdir(LIDAR_PATH)]
    if not args.overwrite:
        print("Do not overwrite clusters.")
        road_frames = [os.path.splitext(filename)[0] for filename in os.listdir(ROAD_MESH_PATH)]
        frame_names = [frame for frame in frame_names if frame not in road_frames]
    for frame_name in sorted(frame_names):
        if counter == args.num_frame:
            break
        else:
            counter += 1
        
        print("Frame {}: {}/{} frames.".format(frame_name, counter, len(frame_names)))
        
        with open(os.path.join(args.road_path, frame_name + ".pkl"), "rb") as f:
            road_points = pickle.load(f)
        road = o3d.geometry.PointCloud()
        road.points = o3d.utility.Vector3dVector(road_points)

        #Waymo loading
        with open(os.path.join(LIDAR_PATH, frame_name + ".pkl"), "rb") as f:
            pcd_dict = pickle.load(f)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_dict["lidars"]["points_xyz"])

        #Front view cropping
        cropped_pcd = front_view(pcd)
        cropped_road = front_view(road)

        #Estimate overall ground (road and other)
        cropped_ground, _  = estimate_ground(cropped_pcd)

        #Improve road prediction
        cropped_road, _ = estimate_ground(cropped_road)
        cropped_road = remove_statistical_outlier(cropped_road)

        #Claculate concave hull of road
        flattend_road_mesh, road_mesh = alpha_surface_reconstruction(cropped_road)
        o3d.io.write_triangle_mesh(os.path.join(ROAD_MESH_PATH, frame_name + ".ply"), road_mesh)