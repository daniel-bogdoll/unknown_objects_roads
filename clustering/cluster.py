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

def compute_signed_distance_and_closest_geometry(scene: o3d.t.geometry.RaycastingScene(), query_points: np.ndarray):
    """
    Filters the points of a point cloud which are not considered as "on the road" by 
    raycasting 

    Args:
        scene (o3d.t.geometry.RaycastingScene): The scene describing the road surface
        query_points (np.ndarray): Point cloud coords to filter

    Returns:
        tuple(np.array, np.ndarray): The point distances and closest points
    """
    closest_points = scene.compute_closest_points(query_points)
    distance = np.linalg.norm(query_points - closest_points['points'].numpy(),
                              axis=-1)
    rays = np.concatenate([query_points, np.ones_like(query_points)], axis=-1)
    intersection_counts = scene.count_intersections(rays).numpy()
    is_inside = intersection_counts % 2 == 1
    distance[is_inside] *= -1
    return distance, closest_points['geometry_ids'].numpy()

def clustering(pcd: o3d.geometry.PointCloud):
    """
    Performs DBSCAN for the point cloud.

    Args:
        pcd (o3d.geometry.PointCloud): Point cloud to cluster

    Returns:
       tuple(o3d.geometry.PointCloud, np.array): Original point cloud with cluster colors and 
       the array of the point's individual cluster
    """
    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        labels = np.array(pcd.cluster_dbscan(eps=1, min_points=30, print_progress=True))

        max_label = labels.max()
        print(f"point cloud has {max_label + 1} clusters")
        colors = plt.get_cmap("tab20b")(labels / (max_label if max_label > 0 else 1))
        colors[labels < 0] = 0
        pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
    return pcd, labels

def plot_boxes(predictions, labels):
    """
    Paints the bounding boxes for the given predictions based on the provided labels. 

    Args:
        predictions (np.ndarray): Bounding box predictions 
        labels (np.array): Predicted labels for the predicted bounding boxes

    Returns:
        list(o3d.geometry.LineSet): Bounding box visuals
    """
    visuals =[] 
    num_det = predictions.shape[0]
    for i in range(num_det):
        box = predictions[i:i+1]#.numpy()
        label = labels[i]
        corner = center_to_corner_box3d(box[:, :3], box[:, 3:6], box[:, -1])[0].tolist()
        color = label2color(int(label) -1)
        visuals.append(corners_to_lines(corner, color))
    return visuals

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo 3D clustering & CenterPoint 3D')
    parser.add_argument("--road_path", type=str, required=True)
    parser.add_argument("--lidar_path", type=str, required=True)
    parser.add_argument("--prediction_path", type=str)
    parser.add_argument('--output_path', type=str, required=True)
    parser.add_argument('--visual', action='store_true', default=False)
    parser.add_argument("--online", action='store_true', default=False)    
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument('--num_frame', default=-1, type=int)
    parser.add_argument("--show_alpha_shape", action='store_true', default=False)
    parser.add_argument("--save_road_mesh", action='store_true', default=False)
    args = parser.parse_args()

    CLUSTER_PATH = os.path.join(args.output_path, 'cluster')

    assert os.path.isdir(args.lidar_path)
    assert os.path.isdir(args.road_path)
    if args.prediction_path:
        assert os.path.isfile(args.prediction_path)
        #Load centerpoint predictions
        with open(args.prediction_path, 'rb') as f:
            detection_preds_dict = pickle.load(f)

    if not os.path.isdir(CLUSTER_PATH):
        os.makedirs(CLUSTER_PATH, exist_ok=True)

    VISUALIZATION_PATH = os.path.join(args.output_path, "visualizations", "lidar")
    if args.visual and not os.path.isdir(VISUALIZATION_PATH):
        os.makedirs(VISUALIZATION_PATH, exist_ok=True)

    if args.save_road_mesh:
        ROAD_MESH_PATH = os.path.join(args.output_path, "road_mesh")
        if not os.path.isdir(ROAD_MESH_PATH):
            os.mkdir(ROAD_MESH_PATH)

    if args.visual:
        vis= o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1280, visible=args.online) #only for saving
        run_vis = True

    counter = 0
    frame_names = [os.path.splitext(filename)[0] for filename in os.listdir(args.lidar_path)]
    if not args.overwrite:
        print("Do not overwrite clusters.")
        clustered_frames = [os.path.splitext(filename)[0] for filename in os.listdir(CLUSTER_PATH)]
        frame_names = [frame for frame in frame_names if frame not in clustered_frames]
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
        with open(os.path.join(args.lidar_path, frame_name + ".pkl"), "rb") as f:
            pcd_dict = pickle.load(f)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_dict["lidars"]["points_xyz"])

        #Front view cropping
        cropped_pcd = front_view(pcd)
        cropped_road = front_view(road)

        #Improve road prediction
        cropped_road, _ = estimate_ground(cropped_road)
        cropped_road = remove_statistical_outlier(cropped_road)

        #Claculate concave hull of road
        flattend_road_mesh, road_mesh = alpha_surface_reconstruction(cropped_road)
        
        if args.save_road_mesh:
            o3d.io.write_triangle_mesh(os.path.join(ROAD_MESH_PATH, frame_name + ".ply"), road_mesh)

        #Filter PC on road mesh
        scene = o3d.t.geometry.RaycastingScene()
        flattend_road_mesh = o3d.t.geometry.TriangleMesh.from_legacy(flattend_road_mesh)
        _ = scene.add_triangles(flattend_road_mesh) 
        distance, closest_points = compute_signed_distance_and_closest_geometry(scene, np.asarray(flatten_pcd(cropped_pcd).points, dtype=np.float32))
        on_road_indices =  np.argwhere(distance <= 0)
        pc_on_road = cropped_pcd.select_by_index(on_road_indices)

        #Remove ground and road of pc on road
        road_ground, pc_without_ground = estimate_ground(pc_on_road, 0.2, num_iterations=300)
        distances = np.asarray(flatten_pcd(pc_without_ground).compute_point_cloud_distance(flatten_pcd(cropped_road)))
        
        pc_without_road = pc_without_ground.select_by_index(np.argwhere((distances <= 0)), invert=True)

        #clustering
        if len(pc_without_road.points) > 0:
            pc_without_road, labels = clustering(pc_without_road)
        else: 
            labels = np.array([])

        #filter out outliers 
        cluster_labels = labels[np.argwhere(labels >= 0)] 
        clustered_obj = pc_without_road.select_by_index(np.argwhere(labels >= 0))

        clusters = np.hstack((np.asarray(clustered_obj.points), cluster_labels))
        labeled_clusters= np.c_[clusters, -np.ones(clusters.shape[0])]

        #Create AABB
        sorted_clusters = clusters[clusters[:,-1].argsort()]
        if sorted_clusters.size == 0:
            cluster_list = []
        else:
            cluster_list = np.split(sorted_clusters[:,:-1], np.unique(sorted_clusters[:,-1], return_index=True)[1][1:])
        cluster_bboxes = [o3d.geometry.AxisAlignedBoundingBox.create_from_points(o3d.utility.Vector3dVector(cluster)) for cluster in cluster_list]
        print("Found {} bboxs in lidar".format(len(cluster_bboxes)))

        cluster_line_sets = [o3d.geometry.LineSet().create_from_axis_aligned_bounding_box(box) for box in cluster_bboxes]
        for lineset in cluster_line_sets:
            lineset.colors = o3d.utility.Vector3dVector([[255, 0,0] for i in range(12)])

        cluster_bboxs_dims =[np.concatenate([cluster_bbox.get_center(), cluster_bbox.get_extent()]) for cluster_bbox in cluster_bboxes]
        if len(cluster_bboxs_dims) >= 2:
            cluster_bboxs_dims = np.vstack(cluster_bboxs_dims)
        else: 
            cluster_bboxs_dims = np.array(cluster_bboxs_dims)

        #initialize labels (as all anomalies, i.e. -1)
        labeled_cluster_bboxs_dims = np.c_[cluster_bboxs_dims, -np.ones(cluster_bboxs_dims.shape[0])]

        #bboxes matching with centerpoint predictions
        if args.prediction_path:
            assert frame_name in detection_preds_dict
            pred_detections = detection_preds_dict[frame_name]
            SCORE_THRESHOLD = 0.5
            score_selected_index = np.argwhere(pred_detections["scores"] > SCORE_THRESHOLD).flatten()
            pred_bboxes_points = pred_detections["box3d_lidar"][score_selected_index]
            pred_labels = pred_detections["label_preds"][score_selected_index]
            if args.visual:
                pred_box_linesets = plot_boxes(pred_bboxes_points, pred_labels)

            pred_AABBs = center_to_AABB_box3d(pred_bboxes_points)
            PIBB_THRESHOLD = 0.5

            for lineset in pred_box_linesets:
                lineset.colors = o3d.utility.Vector3dVector([[0, 0, 0] for i in range(12)])

            matched_cluster_pc_list = []
            for cluster_index, cluster_points in zip(range(len(cluster_list)), cluster_list):    
                cluster_pc = o3d.geometry.PointCloud()
                cluster_pc.points = o3d.utility.Vector3dVector(cluster_points)
                cluster_pc.paint_uniform_color([0.5,0,0]) #init all points red (not covered)
                for AABB_index, AABB in zip(range(len(pred_AABBs)), pred_AABBs):
                    num_points = len(cluster_points)
                    if (num_points) == 0: break
                    points_in_AABB_index = AABB.get_point_indices_within_bounding_box(o3d.utility.Vector3dVector(cluster_points))
                    if((len(points_in_AABB_index) / num_points) >= PIBB_THRESHOLD):
                        labeled_cluster_bboxs_dims[cluster_index, -1] = pred_labels[AABB_index]
                        if args.visual:
                            # colors = np.asarray(cluster_pc.colors)
                            # colors[points_in_AABB_index] = [0, 255, 0]
                            # cluster_pc.colors = o3d.utility.Vector3dVector(colors)
                            cluster_pc.paint_uniform_color([0,1,0])
                            matched_cluster_pc_list.append(cluster_pc)
                            cluster_line_sets[cluster_index].colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(12)]) 
                            pred_box_linesets[AABB_index].colors = o3d.utility.Vector3dVector([[0, 255, 0] for i in range(12)])
                        break
        else:
            clustered_obj.paint_uniform_color([1, 0, 0])

        print(labeled_cluster_bboxs_dims[:,-1])
        with open(os.path.join(CLUSTER_PATH, frame_name + ".pkl"), 'wb') as f:
            pickle.dump({"cluster_pcs": labeled_clusters, "cluster_bboxes": labeled_cluster_bboxs_dims}, f)

        if args.visual:
            # run_vis = True
            pcd.paint_uniform_color([0.8,0.8,0.8])
            cropped_road.paint_uniform_color([0,0,1])
            visuals = [pcd, cropped_road]
            if args.show_alpha_shape:
                road_mesh.paint_uniform_color([0,0,0.5])
                visuals += [road_mesh]
            if args.prediction_path:
                visuals += matched_cluster_pc_list
            else:
                visuals += [clustered_obj]
            visuals += cluster_line_sets

            if args.prediction_path:
                #Visualize center point cluster matching
                visuals += pred_box_linesets #TODO delete
            
            for visual in visuals:
                vis.add_geometry(visual)
            ctr = vis.get_view_control()
            ctr.set_front([-1, 0, 0])
            ctr.set_lookat([4.839074535686799, 1.18155124454753, 5.2192577163444129])
            ctr.set_up([0.0, 0.0, 1])
            ctr.set_zoom(0.03)
            ctr.change_field_of_view(step=-5)

            # Updates
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
            time.sleep(0.001)
            vis.capture_screen_image(os.path.join(VISUALIZATION_PATH, "{}.png".format(frame_name)), do_render=True)

            # Remove previous geometry
            vis.clear_geometries()

    if args.visual:
        vis.destroy_window()
        # del ctr
        del vis