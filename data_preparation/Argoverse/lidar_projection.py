"""
Conveniant script to produce lidar-projected images from Argoverse tracking dataset

Assumes that argoverse-api is installed: https://github.com/argoai/argoverse-api
"""

import click
from pathlib import Path
import numpy as np
import json
from tqdm import  tqdm
from numpy import savez_compressed

from argoverse.utils.camera_stats import (
    CAMERA_LIST,
    RING_CAMERA_LIST,
    STEREO_CAMERA_LIST,
    get_image_dims_for_camera
)

from argoverse.utils.calibration import (
    project_lidar_to_img,
    point_cloud_to_homogeneous
)

from argoverse.utils.ply_loader import load_ply



def validate_camera_option(cameras: str):
    """
    Just checks the list of cameras names is valid and return a list without duplicates
    Assumes `cameras` is a comma separated list of camera names
    """
    camera_set = set()

    cameras = cameras.split(',')
    for camera in cameras:

        if camera == "ring":
            camera_set.update(set(RING_CAMERA_LIST)) # inplace union
        elif camera == "stereo":
            camera_set.update(set(STEREO_CAMERA_LIST))
        elif camera in CAMERA_LIST:
            camera_set.add(camera)
        else:
            raise ValueError(f"Camera of name {camera} is not valid. Cameras available: {CAMERA_LIST}")

    return list(camera_set)

def project_and_save(lidar_filepath, output_base_path, calib_data, cameras):

    pts = load_ply(lidar_filepath)  # point cloud, numpy array Nx3 -> N 3D coords

    points_h = point_cloud_to_homogeneous(pts).T

    for camera_name in cameras:
        uv, uv_cam, valid_pts_bool = project_lidar_to_img(points_h, calib_data, camera_name)
        uv = uv[valid_pts_bool].astype(np.int32)  # Nx2 projected coords in pixels
        uv_cam = uv_cam.T[valid_pts_bool]  # Nx3 projected coords in meters

        img_width, img_height = get_image_dims_for_camera(camera_name)
        img = np.zeros((img_height, img_width))
        img[uv[:, 1], uv[:, 0]] = uv_cam[:, 2]  # image of projected lidar measurements

        lidar_filename = lidar_filepath.stem
        output_dir = output_base_path / camera_name
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (lidar_filename + ".npz")
        savez_compressed(str(output_path), img)
    return None

@click.command()
@click.argument('argo_tracking_root_dir', type=click.Path(exists=True, file_okay=False)) # .../argoverse-tracking/ under which you can find "train1", "train2", ...
@click.argument('output_dir', type=click.Path(file_okay=False))
@click.option('--cameras', type=str) # comma-separated list of camera names or "ring" or "stereo"
@click.option('--ip_basic', is_flag=True)
@click.option('--acc_sweeps', is_flag=True)
def main(argo_tracking_root_dir, output_dir, cameras, ip_basic, acc_sweeps):
    print('Preprocessing data....')
    print("INPUT DIR: ", argo_tracking_root_dir)
    print("OUTPUT DIR: ", output_dir)

    if cameras is None:
        cameras = CAMERA_LIST
    else:
        cameras = validate_camera_option(cameras)

    print(cameras)

    argo_tracking_root_dir = Path(argo_tracking_root_dir).expanduser()
    output_dir = Path(output_dir).expanduser()

    train_namedirs = ["train1", "train2", "train3", "train4", "test", "val"]
    train_dirs = [argo_tracking_root_dir / train_namedir for train_namedir in train_namedirs]

    for train_dir in train_dirs:
        log_dirs = sorted(list(train_dir.iterdir()))
        for log_dir in log_dirs:
            lidar_dir = log_dir / "lidar"
            lidar_filepaths = [f for f in sorted(list(lidar_dir.iterdir()))]
            total = len(lidar_filepaths)

            calib_filepath = str(log_dir / "vehicle_calibration_info.json")
            output_base_path = output_dir / train_dir.stem / log_dir.stem

            with open(calib_filepath, "r") as f:
                calib_data = json.load(f)
            args = (output_base_path, calib_data, cameras)

            with tqdm(lidar_filepaths, desc=f"{train_dir.stem} | log {log_dir.stem} ", total=total) as progress:
                for lidar_filepath in progress:
                    project_and_save(lidar_filepath, *args)

    print('Preprocessing of LiDAR data Finished.')

if __name__ == '__main__':
    main()