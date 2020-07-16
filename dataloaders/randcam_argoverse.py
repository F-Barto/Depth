"""
This module implements a Dataloader suitable for the self-supervisied training of a monocular depth estimation method on
the argoverse Tracking dataset (please cite their work and our work if you use it).

assumes that the argoverse API is installed

In this module, the docstring follows the NumPy/SciPy formatting rules.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from torch.utils.data import Dataset
from random import randrange
import skimage.io
import pickle

from pytorch_lightning import _logger as terminal_logger


VEHICLE_CALIBRATION_INFO_FILENAME = 'vehicle_calibration_info.json'



class RandCamSequentialArgoverseLoader(Dataset):
    """
    argoverse Tracking Dataloader that loads:
        - the image at time t (the target view)
        - the neighbouring images, e.g., at time t-1 and t+1 (the source views)
        - the depth image from the LiDAR 3D data
        - the pose

    Attributes
    ----------
    says_str : str
        a formatted string to print out what the animal says

    Methods
    -------
    says(sound=None)
        Prints the animals name and what sound it makes
    """

    def __init__(self, argoverse_tracking_root_dir,  gt_depth_root_dir=None,
                 sparse_depth_root_dir=None, data_transform=None, data_transform_options=None,
                 load_pose=False, split_file=None, input_channels=3, fix_cam_idx=None):

        """
        Parameters
        ----------
        argoverse_tracking_root_dir : str
            The path to the root of the argoverse tracking dataset,
            e.g., /home/clear/fbartocc/data/ARGOVERSE/argoverse-tracking
        gt_depth_root_dir : str
            The path to where the computed depth maps are stored.
            If not None, the depth from LiDAR data of each frame will be returned.
        data_transform :
            Transform applied to each data sample before returning it.
        source_views_indexes : list of int
            The relative indexes to sample from the neighbouring views of the target view.
            It is expected that the list is in ascending order and does not contains 0 (corresponding to the target).
            For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1
        load_pose : bool
            If True, the pose of each frame will be returned.
        """
        super().__init__()

        self.load_pose = load_pose

        argoverse_tracking_root_dir = Path(argoverse_tracking_root_dir).expanduser()
        assert argoverse_tracking_root_dir.exists(), argoverse_tracking_root_dir
        self.argoverse_tracking_root_dir = argoverse_tracking_root_dir

        assert input_channels in [1,3], input_channels
        self.input_channels = {1: 'gray', 3: 'rgb'}[input_channels]

        self.data_transform = data_transform
        self.data_transform_options = data_transform_options

        if data_transform is not None:
            assert data_transform_options is not None

        if gt_depth_root_dir is not None:
            gt_depth_root_dir = Path(gt_depth_root_dir)
            assert gt_depth_root_dir.exists(), gt_depth_root_dir
            self.gt_depth_root_dir = gt_depth_root_dir
            terminal_logger.info(f"The GT depth from LiDAR data of each frame will be loaded from {str(gt_depth_root_dir)}.")

        self.load_sparse_depth = False
        if sparse_depth_root_dir is not None:
            sparse_depth_root_dir = Path(sparse_depth_root_dir)
            assert sparse_depth_root_dir.exists(), sparse_depth_root_dir
            self.sparse_depth_root_dir = sparse_depth_root_dir
            self.load_sparse_depth = True
            terminal_logger.info(
                f"The sparse depth from LiDAR data of each frame will be loaded from {str(sparse_depth_root_dir)}.")

        assert split_file is not None
        with open(split_file, 'rb') as f:
            split_data = pickle.load(f)
        source_views_indexes = split_data['source_views_indexes']
        self.split_name = split_data['split_name']
        self.camera_list = split_data['camera_list']
        self.camera_configs = split_data['camera_configs']
        self.samples_paths = split_data['samples_paths']
        self.translation_magnitudes = split_data.get('translation_magnitudes', None)

        if fix_cam_idx is not None:
            assert fix_cam_idx < len(self.camera_list), f"{len(self.camera_list)} cameras | {self.camera_list}"
            self.fix_cam_idx = fix_cam_idx


        assert self.split_name in ['train', 'val', 'test']

        self.source_views_requested = source_views_indexes is not None and len(source_views_indexes) > 0

        # filter out samples where gt depth is not available for val and test
        if self.split_name in ['val', 'test']:
            terminal_logger.info("Filtering out samples which don't have ground truth depth")
            self.samples_paths = [sample for sample in self.samples_paths if self.has_gt_depth(sample)]

        terminal_logger.info(f'Dataset for split {self.split_name} ready.\n\n' + '-'*90 + '\n\n')

    def has_gt_depth(self, sample):
        lidar_path = sample[0]
        # we only need to test for one camera, as the only case where gt is not available
        # is when we are at log boundaries so we can't accumulate lidar sweeps. So gt is not available for all cams.
        camera_name = self.camera_list[0]
        gt_depth_filepath = self.get_projected_lidar_path(camera_name, lidar_path, self.gt_depth_root_dir)

        return Path(gt_depth_filepath).exists()


    def get_projected_lidar_path(self, camera_name, lidar_path, depth_base_dir):
        lidar_path = Path(lidar_path)
        split = lidar_path.parents[2].stem
        log = lidar_path.parents[1].stem
        return str(depth_base_dir / split / log / camera_name / (lidar_path.stem + '.npz'))

    def read_tiff_depth(self, file_path):
        depth = skimage.io.imread(file_path).astype(np.float32)
        return np.expand_dims(depth, axis=2)

    def read_npz_depth(self, file):
        depth = np.load(file)['arr_0'].astype(np.float32)
        return np.expand_dims(depth, axis=2)

    def load_img(self, file_path):
        img = Image.open(file_path)
        if self.input_channels == 'gray':
            return img.convert('L')
        return img

    def __len__ (self):
        return len(self.samples_paths)


    def __getitem__(self, idx):

        if self.fix_cam_idx is not None:
            cam_idx = self.fix_cam_idx
        else:
            if self.split_name == 'train':
                cam_idx = randrange(len(self.camera_list))
            else:
                cam_idx = idx % len(self.camera_list)

        camera_name = self.camera_list[cam_idx]

        # self.samples_paths[idx][0] is lidar, subsequent indexes are cameras
        target_view_path = self.argoverse_tracking_root_dir / self.samples_paths[idx][1+cam_idx][0]
        target_view = self.load_img(target_view_path)

        sample = {
            'target_view': target_view,
            'idx': idx
        }

        lidar_path = self.samples_paths[idx][0]  # the .ply file

        if self.split_name in ['val', 'test']:
            projected_lidar_path = self.get_projected_lidar_path(camera_name, lidar_path, self.gt_depth_root_dir)
            projected_lidar = self.read_npz_depth(projected_lidar_path)
            sample['projected_lidar'] = projected_lidar

        if self.load_sparse_depth:
            sparse_projected_lidar_path = self.get_projected_lidar_path(camera_name, lidar_path,
                                                                        self.sparse_depth_root_dir)
            sparse_projected_lidar = self.read_npz_depth(sparse_projected_lidar_path)
            sample['sparse_projected_lidar'] = sparse_projected_lidar

        if self.source_views_requested:
            source_views_paths = [str(self.argoverse_tracking_root_dir / p)
                                  for p in self.samples_paths[idx][1 + cam_idx][1]]
            source_views = [self.load_img(p) for p in source_views_paths]
            sample['source_views'] = source_views

        if self.translation_magnitudes is not None and self.load_pose:
            sample['translation_magnitudes'] = self.translation_magnitudes[idx][cam_idx]

        split = target_view_path.parents[2].stem
        log = target_view_path.parents[1].stem
        camera_name = target_view_path.parent.stem
        image_timestamp = Path(target_view_path).stem
        sample['filename'] = f"{split}_{log}_{image_timestamp}"

        cam_config = self.camera_configs[log][camera_name]
        sample['intrinsics'] = cam_config.intrinsic[:3, :3]

        if self.data_transform is not None:
            self.data_transform(sample, **self.data_transform_options)

        return sample