"""
Conveniant script to produce lidar-projected images from argoverse tracking dataset

Assumes that argoverse-api is installed: https://github.com/argoai/argoverse-api
"""

import click
from pathlib import Path
import numpy as np
import json
from tqdm import  tqdm
from numpy import savez_compressed

from argoverse.utils.camera_stats import CAMERA_LIST, get_image_dims_for_camera
from argoverse.utils.calibration import project_lidar_to_img_motion_compensated, point_cloud_to_homogeneous
from argoverse.data_loading.synchronization_database import SynchronizationDB
from argoverse.utils.ply_loader import load_ply

from .common import validate_camera_option
from .data_synchronization import get_synchronized_data


def get_neighbouring_lidar_timestamps(db, log, initial_timestamp, neighbours_indexes):
    """
    :param db: argoverse SynchronizationDB object
    :param log: string, log name
    :param initial_timestamp: the timestamp from which we want the neighbouring lidar timestamps
    :param neighbours_indexes: a list of neighbouring frames indexes (int), e.g., [-1,0,1]
    :return:
    """

    if 0 not in neighbours_indexes:
        neighbours_indexes.append(0)

    neighbours_indexes = list(set(neighbours_indexes)) # discard duplicates
    neighbours_indexes.sort()

    lidar_timestamps = db.per_log_lidartimestamps_index[log]
    timestamp_idx = np.searchsorted(lidar_timestamps, initial_timestamp)

    neighbours_timestamps = []
    for neighbour_index in neighbours_indexes:
        neighbour_timestamp = lidar_timestamps[timestamp_idx + neighbour_index]
        neighbours_timestamps.append(neighbour_timestamp)

    return neighbours_timestamps

def project_and_save(lidar_filepath, output_base_path, calib_data, cameras, db, acc_sweeps, ip_basic):

    log = lidar_filepath.parents[1].stem
    lidar_timestamp = lidar_filepath.stem[3:]

    neighbouring_timestamps = get_neighbouring_lidar_timestamps(db, log, lidar_timestamp, acc_sweeps)

    pts = load_ply(lidar_filepath)  # point cloud, numpy array Nx3 -> N 3D coords

    points_h = point_cloud_to_homogeneous(pts).T

    uv, uv_cam, valid_pts_bool = project_lidar_to_img_motion_compensated(
        points_h,  # these are recorded at lidar_time
        copy.deepcopy(calib),
        camera_name,
        int(cam_timestamp),
        int(lidar_timestamp),
        str(split_dir),
        log,
    )


    for camera_name in cameras:
        uv, uv_cam, valid_pts_bool = project_lidar_to_img_motion_compensated(points_h, calib_data, camera_name)
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
@click.option('--acc_sweeps', type=int)
@click.option('--ip_basic', is_flag=True)
def main(argo_tracking_root_dir, output_dir, cameras, acc_sweeps, ip_basic):
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

    split_namedirs = ["train1", "train2", "train3", "train4", "test", "val"]
    split_dirs = [argo_tracking_root_dir / split_namedir for split_namedir in split_namedirs]

    for split_dir in split_dirs:

        db = SynchronizationDB(str(split_dir))

        log_dirs = sorted(list(split_dir.iterdir()))
        for log_dir in log_dirs:
            lidar_dir = log_dir / "lidar"
            lidar_filepaths = [f for f in sorted(list(lidar_dir.iterdir()))]
            total = len(lidar_filepaths)

            calib_filepath = str(log_dir / "vehicle_calibration_info.json")
            output_base_path = output_dir / split_dir.stem / log_dir.stem

            with open(calib_filepath, "r") as f:
                calib_data = json.load(f)
            args = (output_base_path, calib_data, cameras, db, acc_sweeps, ip_basic)

            with tqdm(lidar_filepaths, desc=f"{split_dir.stem} | log {log_dir.stem} ", total=total) as progress:
                for lidar_filepath in progress:
                    project_and_save(lidar_filepath, *args)

    print('Preprocessing of LiDAR data Finished.')

if __name__ == '__main__':
    main()