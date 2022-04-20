import numpy as np
import cv2 as cv2
import argparse, os
from datetime import datetime

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Pipeline visualization')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument("--online", action='store_true', default=False)    
    parser.add_argument("--overwrite", action='store_true', default=False)
    parser.add_argument('--num_frame', default=-1, type=int)
    args = parser.parse_args()

    MSEG_PATH = os.path.join(args.root_path,"mseg")
    ROAD_MESH_PATH = os.path.join(args.root_path,"road_mesh")
    LIDAR_PATH = os.path.join(args.root_path,"lidar")
    CAMERA_PATH = os.path.join(args.root_path,"camera")

    assert os.path.isdir(MSEG_PATH)
    assert os.path.isdir(ROAD_MESH_PATH)
    assert os.path.isdir(LIDAR_PATH)
    assert os.path.isdir(CAMERA_PATH)

    PIPELINE_PATH = os.path.join(args.root_path,"pipeline")
    if not os.path.isdir(PIPELINE_PATH):
        os.mkdir(PIPELINE_PATH)
    fourcc = cv2.VideoWriter_fourcc(*'MP4V')
    out = cv2.VideoWriter(os.path.join(PIPELINE_PATH, 'pipeline_vis-{}.mp4'.format(str(datetime.now()))), fourcc, 30.0, (1920,1280))

    counter = 0
    frame_names = [os.path.splitext(filename)[0] for filename in os.listdir(LIDAR_PATH)]
    if not args.overwrite:
        print("Do not overwrite clusters.")
        pipeline_frames = [os.path.splitext(filename)[0] for filename in os.listdir(PIPELINE_PATH)]
        frame_names = [frame for frame in frame_names if frame not in  pipeline_frames]
    for frame_name in sorted(frame_names):
        if counter == args.num_frame:
            break
        else:
            counter += 1
        
        print("Frame {}: {}/{} frames.".format(frame_name, counter, args.num_frame if args.num_frame else len(frame_names)))

        mseg_img = cv2.imread(os.path.join(MSEG_PATH, '{}.png'.format(frame_name)))
        road_mesh_img = cv2.imread(os.path.join(ROAD_MESH_PATH, '{}.png'.format(frame_name)))
        lidar_img = cv2.imread(os.path.join(LIDAR_PATH, '{}.png'.format(frame_name)))
        camera_img = cv2.imread(os.path.join(CAMERA_PATH, '{}.jpg'.format(frame_name)))

        #resize to fit
        dim = (int(1920 / 2), int(1280 / 2))
        mseg_img_resized = cv2.resize(mseg_img,dim, interpolation = cv2.INTER_AREA)
        road_mesh_img_resized = cv2.resize(road_mesh_img, dim, interpolation = cv2.INTER_AREA)
        lidar_img_resized = cv2.resize(lidar_img, dim, interpolation = cv2.INTER_AREA)
        camera_img_resized = cv2.resize(camera_img, dim, interpolation = cv2.INTER_AREA)

        def opaque_rectangle(img, x, y, w, h):
            sub_img = img[y:y+h, x:x+w]
            white_rect = np.zeros(sub_img.shape, dtype=np.uint8)
            res = cv2.addWeighted(sub_img, 0.5, white_rect, 0.5, 1.0)
            img[y:y+h, x:x+w] = res
            return img

        x,y,w,h = 0,0,400,80
        # Create background rectangle with color
        #cv2.rectangle(mseg_img_resized, (x,x), (x + w, y + h), (0,0,0), -1)
        mseg_img_resized = opaque_rectangle(mseg_img_resized, x,y, w, h)
        cv2.putText(mseg_img_resized, "MSeg road segmentation", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
        cv2.putText(road_mesh_img_resized, "Alpha shape road", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (255,0,0), 2)
        cv2.putText(road_mesh_img_resized, "DBSCAN clusters", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
        cv2.putText(lidar_img_resized, "CenterPoint 3D detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,255,0), 2)
        cv2.putText(lidar_img_resized, "3D anomaly", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)
        camera_img_resized = opaque_rectangle(camera_img_resized, x,y, 300, 110)
        cv2.putText(camera_img_resized, "CLIP 2D detection", (40, 40), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,255,0), 2)
        cv2.putText(camera_img_resized, "3D & 2D anomaly", (40, 70), cv2.FONT_HERSHEY_SIMPLEX, .75, (0,0,255), 2)

        first_row = np.concatenate((mseg_img_resized, road_mesh_img_resized), axis=1)
        second_row = np.concatenate((lidar_img_resized, camera_img_resized), axis=1)
        pipeline_img = np.concatenate((first_row, second_row), axis=0)
        if args.online:
            cv2.imshow('Anomaly detection pipeline', pipeline_img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        out.write(pipeline_img)

    if args.online: 
        cv2.destroyAllWindows()
    out.release()