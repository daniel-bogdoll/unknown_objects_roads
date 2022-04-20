import numpy as np
import open3d as o3d
import pickle
import os
import matplotlib.pyplot as plt
import time
import argparse

import sys 
from os import path
sys.path.append(path.relpath('./'))
sys.path.append(path.relpath('./CenterPoint'))
from CenterPoint.tools.visual import center_to_corner_box3d, label2color, corners_to_lines

def center_to_AABB_box3d(boxes):
    """
    Converts center bounding boxes to axis-aligned bounding boxes.

    Args:
        boxes (np.ndarray): Center based bounding boxes coords

    Returns:
        np.ndarray: Axis-aligned bounding boxes
    """
    AABBs =[] 
    num_det = len(boxes)
    for i in range(num_det):
        box = boxes[i:i+1]
        corner = np.array(center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist())
        AABBs.append(o3d.geometry.AxisAlignedBoundingBox(corner.min(axis=0), corner.max(axis=0)))
    return AABBs

def plot_boxes(predictions, score_thresh):
    """
    Plots the bounding boxes for the given predictions and the given confidence threshold

    Args:
        predictions (np.ndarray): Predicted bounding box coords
        score_thresh (double): Confidence threshold for bounding boxes

    Returns:
        list(o3d.geometry.LineSet): Bounding box visuals
    """
    visuals =[] 
    num_det = predictions['scores'].shape[0]
    for i in range(num_det):
        score = predictions['scores'][i]
        if score < score_thresh:
            continue 

        box = predictions['box3d_lidar'][i:i+1]#.numpy()
        label = predictions['label_preds'][i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(int(label) -1)
        visuals.append(corners_to_lines(corner, color))
    return visuals

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo 3D clustering')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument("--online", action='store_true', default=False)    
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    CLUSTER_PATH = os.path.join(args.root_path,'cluster')
    LIDAR_PATH = os.path.join(args.root_path, 'lidar')
    ROAD_PATH = os.path.join(args.root_path, 'road_mask')

    assert os.path.isdir(LIDAR_PATH)
    assert os.path.isdir(CLUSTER_PATH)
    assert os.path.isdir(ROAD_PATH)
    assert os.path.isfile(args.prediction_path)

    if not os.path.isdir(CLUSTER_PATH):
        os.mkdir(CLUSTER_PATH)
    
    VISUALIZATION_PATH = os.path.join(args.root_path, "visualizations", "lidar")
    if not os.path.isdir(VISUALIZATION_PATH):
        os.mkdir(VISUALIZATION_PATH)

    with open(args.prediction_path, 'rb') as f:
        detection_preds_dict = pickle.load(f)

    counter = 0
    frame_names = [os.path.splitext(filename)[0] for filename in os.listdir(LIDAR_PATH)]
    if not args.overwrite:
        print("Do not overwrite visualizations.")
        visualized_frames = [os.path.splitext(filename)[0] for filename in os.listdir(VISUALIZATION_PATH)]
        frame_names = [frame for frame in frame_names if frame not in visualized_frames]

    vis= o3d.visualization.Visualizer()
    vis.create_window(width=1920, height=1280, visible=args.online) #only for saving
    
    for frame_name in sorted(frame_names):
        if counter == args.num_frame:
            break
        else:
            counter += 1
                    
        pred_detections = detection_preds_dict[frame_name]

        pc_name = path.join(LIDAR_PATH, frame_name +".pkl")
        with open(pc_name, 'rb') as f:
            lidar_dict = pickle.load(f)


        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(lidar_dict["lidars"]["points_xyz"])
        pcd.paint_uniform_color([0.8,0.8,0.8])


        cluster_name = path.join(CLUSTER_PATH, frame_name +".pkl")
        with open(cluster_name, 'rb') as f:
            cluster_dict = pickle.load(f) #x,y,z,cluster,label
        labeled_clusters = cluster_dict["cluster_pcs"]
        clusters = np.delete(labeled_clusters, -1, axis=1)

        with open(path.join(ROAD_PATH, frame_name + ".pkl"), "rb") as f:
            road_points = pickle.load(f)
            road = o3d.geometry.PointCloud()
            road.points = o3d.utility.Vector3dVector(road_points)


        sorted_clusters = clusters[clusters[:,-1].argsort()]
        if sorted_clusters.size == 0:
            cluster_list = []
        else:
            cluster_list = np.split(sorted_clusters[:,:-1], np.unique(sorted_clusters[:,-1], return_index=True)[1][1:])

        cluster_bboxes = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster)) for cluster in cluster_list]

        cluster_line_sets = [o3d.geometry.LineSet().create_from_axis_aligned_bounding_box(box) for box in cluster_bboxes]
        for lineset in cluster_line_sets:
            lineset.colors = o3d.utility.Vector3dVector([[255, 0,0] for i in range(12)])

        clusters_pc = o3d.geometry.PointCloud()
        clusters_pc.points = o3d.utility.Vector3dVector(clusters[:,:-1])
        cluster_labels = clusters[:,-1]
        if cluster_labels.size == 0:
            max_label = -1
        else:
            max_label = cluster_labels.max()
        print(f"point cloud has {int(max_label + 1)} clusters")
        colors = plt.get_cmap("tab20b")(cluster_labels / (max_label if max_label > 0 else 1))
        colors[cluster_labels < 0] = 0
        clusters_pc.colors = o3d.utility.Vector3dVector(colors[:, :3])

        SCORE_THRESHOLD = 0.5

        # visuals = [pcd, clusters_pc]
        # visuals += cluster_line_sets
        pred_box_linesets = plot_boxes(pred_detections, SCORE_THRESHOLD)
        # visuals += pred_box_linesets
        # o3d.visualization.draw_geometries(visuals, width=3072, height=1920)


        pred_detections["box3d_lidar"].shape

        pred_bboxes_points = pred_detections["box3d_lidar"][np.argwhere(pred_detections["scores"] > SCORE_THRESHOLD).flatten()]

        AABBs = center_to_AABB_box3d(pred_bboxes_points)

        PIBB_THRESHOLD = 0.5

        for lineset in pred_box_linesets:
            lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(12)])
        
        start = time.time()
        cluster_pc_list = []
        for cluster_index, cluster_points in zip(range(len(cluster_list)), cluster_list):    
            cluster_pc = o3d.geometry.PointCloud()
            cluster_pc.points = o3d.utility.Vector3dVector(cluster_points)
            cluster_pc.paint_uniform_color([0.5,0,0])
            for AABB_index, AABB in zip(range(len(AABBs)), AABBs):
                num_points = len(cluster_points)
                if (num_points) == 0: break
                points_in_AABB_index = AABB.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(cluster_points))
                if((len(points_in_AABB_index) / num_points) >= PIBB_THRESHOLD):
                    # colors = np.asarray(cluster_pc.colors)
                    # colors[points_in_AABB_index] = [0, 255, 0]
                    # cluster_pc.colors = o3d.utility.Vector3dVector(colors)
                    cluster_pc.paint_uniform_color([0,1,0])
                    # cluster_pc_list.append(cluster_pc)
                    
                    pred_box_linesets[AABB_index].colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(12)])
                    cluster_line_sets[cluster_index].colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(12)])
                    break
            cluster_pc_list.append(cluster_pc)

        end = time.time()
        print("Took {} seconds".format(end - start))

        road.paint_uniform_color([0,0,1])
        visuals = [pcd, road]
        visuals += cluster_line_sets
        visuals += pred_box_linesets
        visuals += cluster_pc_list
        for visual in visuals:
                vis.add_geometry(visual)

        ctr = vis.get_view_control()
        ctr.set_front([-1, 0, 0])
        ctr.set_lookat([4.839074535686799, 1.18155124454753, 5.2192577163444129])
        ctr.set_up([0.0, 0.0, 1])
        ctr.set_zoom(0.03)
        ctr.change_field_of_view(step=-5)

        vis.update_geometry(pcd)
        vis.poll_events()
        vis.update_renderer()
        time.sleep(0.001)
        vis.capture_screen_image(os.path.join(VISUALIZATION_PATH, "{}.png".format(frame_name)), do_render=True)

        # Remove previous geometry
        vis.clear_geometries()

    vis.destroy_window()
    del vis
        



