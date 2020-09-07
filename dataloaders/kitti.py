"""
This module implements a Dataloader suitable for the self-supervisied training of a monocular depth estimation method on
the KITTI Raw dataset [1] (please cite their work and our work if you use it).


[1] Andreas Geiger et al., Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite, CVPR 2012
(http://www.cvlibs.net/datasets/kitti/index.php)
@INPROCEEDINGS{Geiger2012CVPR,
  author = {Andreas Geiger and Philip Lenz and Raquel Urtasun},
  title = {Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite},
  booktitle = {Conference on Computer Vision and Pattern Recognition (CVPR)},
  year = {2012}
}


In this module, the docstring follows the NumPy/SciPy formatting rules.
"""


import numpy as np
from PIL import Image
from pathlib import Path

from torch.utils.data import Dataset
from pytorch_lightning import _logger as terminal_logger


KITTI_RAW_LEFT_STEREO_IMAGE_DIR = 'image_02/data'
PROJECTED_VELODYNE_DIR = 'proj_depth/velodyne/image_02'
PROJECTED_GROUNDTRUTH_DIR = 'proj_depth/groundtruth/image_02'
INTRINSICS_MATRIX = 'P_rect_02'
CAMERA_CALIBRATION_FILE_NAME = 'calib_cam_to_cam.txt'


class SequentialKittiLoader(Dataset):
    """
    KITTI Dataloader that loads:
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

    def __init__(self, kitti_root_dir, split_file_path, gt_depth_root_dir=None, sparse_depth_root_dir=None,
                 data_transform=None, data_transform_options=None, source_views_indexes=[-1, 1], load_pose=True,
                 input_channels=3):

        """
        Parameters
        ----------
        kitti_root_dir : str
            The path to the root of the KITTI Raw dataset, <path to KITTI Raw>/<date>
            e.g., /home/clear/fbartocc/data/KITTI_raw/2011_09_26
        split_file_path : str
            The path to a .txt file containing the list the relative path to each image w.r.t. root dir.
            Under the root dir, KITTI raw is organized as follow:
            {capture_date}/{capture_date}_drive_{sequence_idx:04d}_sync/image_02/data/{frame_idx:010d}.png
            e.g., 2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
            with:
            - capture_date = 2011_09_26
            - sequence_idx = 48
            - frame_idx = 85
            Note: we only use the image from the front left camera (camera 2 in KITTI)
        depth_root_dir : str
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

        assert Path(kitti_root_dir).exists, kitti_root_dir
        assert input_channels in [1,3]
        self.input_channels = {1: 'gray', 3: 'rgb'}[input_channels]

        self.data_transform = data_transform
        self.data_transform_options = data_transform_options

        if data_transform is not None:
            assert data_transform_options is not None

        if gt_depth_root_dir is not None:
            gt_depth_root_dir = Path(gt_depth_root_dir)
            assert gt_depth_root_dir.exists(), gt_depth_root_dir
            self.gt_depth_root_dir = gt_depth_root_dir
            terminal_logger.info(
                f"The GT depth from LiDAR data of each frame will be loaded from {str(gt_depth_root_dir)}.")

        self.load_sparse_depth = False
        if sparse_depth_root_dir is not None:
            sparse_depth_root_dir = Path(sparse_depth_root_dir)
            assert sparse_depth_root_dir.exists(), sparse_depth_root_dir
            self.sparse_depth_root_dir = sparse_depth_root_dir
            self.load_sparse_depth = True
            terminal_logger.info(
                f"The sparse depth from LiDAR data of each frame will be loaded from {str(sparse_depth_root_dir)}.")

        src_indexes_err_msg = "It is expected the source index list is in ascending order and does not contains 0 " \
                              "(corresponding to the target)\n " \
                              "For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1\n" \
                              f"yours: {source_views_indexes}"
        assert 0 not in source_views_indexes, src_indexes_err_msg

        self.source_views_requested = len(source_views_indexes) > 0

        self.split_name = Path(split_file_path).stem # used in __getitem__

        self.kitti_root_dir = Path(kitti_root_dir).expanduser()
        self.load_pose = load_pose

        self.intrinsics = self.get_intrinsics_for_all_sequences(split_file_path)
        img_paths = self.load_absolute_paths_from_split_file(split_file_path)

        if self.source_views_requested:
            self.sequence_lengths = self.get_sequences_lengths(split_file_path)
            self.samples_and_source_views_paths = self.get_valid_samples_and_source_views_paths(img_paths,
                                                                                                source_views_indexes)
        else:
            self.samples_and_source_views_paths = [(p, None) for p in img_paths]

        terminal_logger.info(f'Dataset for split {self.split_name} ready.\n\n' + '-'*90 + '\n\n')

    def __len__ (self):
        return len(self.samples_and_source_views_paths)

    def load_absolute_paths_from_split_file(self, split_file_path):
        """
        Method to list the absolute paths to the data samples. Uses the given dataset root path and a split_file
        containing the relative paths  w.r.t. root dir.

        Note: we always load the image from the front left camera (camera 2 in KITTI)

        Parameters
        ----------
        split_file_path: str
            The path to a .txt file containing the list the relative path to each image w.r.t. root dir.

        Returns
        -------
        list
            A list of absolute path that can be used to load the corresponding data sample
        """
        with Path(split_file_path).open() as split_file:
            relative_paths = split_file.readlines()

        img_paths = []
        for i, relative_path in enumerate(relative_paths):
            img_path = self.kitti_root_dir / relative_path.split()[0]

            img_paths.append(str(img_path))

        terminal_logger.info(f'{len(img_paths)} listed files in the split file {self.split_name}.')

        return img_paths

    def get_sequences_lengths(self, split_file_path):
        """
        Returns a dictionary giving the number of frames for each sequences in split_file

        Parameters
        ----------
        split_file_path: str
            The path to a .txt file containing the list the relative path to each image w.r.t. root dir.

        Returns
        -------
        dict
            A dictionary where the key is the directory name of a sequence and the value the number of frames it
             contains, i.e., the lenth of the sequence
        """

        # Under the root dir, KITTI raw is organized as follow
        # {capture_date}/{capture_date}_drive_{sequence_idx:04d}_sync/image_02/data/{frame_idx:010d}.png
        # e.g., 2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
        # thus, each line of the split file is assumed to have the same formatting
        arr = np.genfromtxt(split_file_path, delimiter='/', dtype=str)
        # each row of arr is of the form
        # ['2011_09_26', '2011_09_26_drive_0048_sync', 'image_02', 'data', '0000000085.png']
        used_sequences = np.unique(arr[:, 1])

        sequence_lengths = {}
        for used_sequence in used_sequences:
            capture_date = used_sequence[:10]
            sequence_dir = self.kitti_root_dir / capture_date / used_sequence / KITTI_RAW_LEFT_STEREO_IMAGE_DIR
            assert sequence_dir.exists(), sequence_dir
            sequence_lengths[used_sequence] = len([f for f in sequence_dir.iterdir()])

        return sequence_lengths

    def get_valid_samples_and_source_views_paths(self, img_paths, source_views_indexes):
        """
        Discard the samples that don't have avaible source views, where the source views are defined by source_indexes.

        Parameters
        ----------
        img_paths: list of str
            A list of absolute paths to the data samples.
            Each element of img_paths is of the form:
            path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png

        Returns
        -------
        list of tuple
            A curated list of absolute paths to the data samples ensured to have source views.
            each element of the list is a tuple (target_img_path, [source_view1_img_path, ..., source_viewN_img_path])
        """

        terminal_logger.info(f'Checking if the files given in the split file {self.split_name} '
                             f'exists and have source views...')

        img_paths_and_source_views = []

        for img_path in img_paths:

            # img_path == path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
            img_path = Path(img_path)
            assert img_path.exists(), img_path

            frame_idx = self.get_frame_idx_from_image_path(img_path) # == 85
            sequence_dir = img_path.parents[2].name # == 2011_09_26_drive_0048_sync

            # checks if any source source view index falls out of the sequence's bounds
            sequence_length = self.sequence_lengths[sequence_dir]
            if frame_idx + source_views_indexes[0] < 0 or frame_idx + source_views_indexes[-1] >= sequence_length:
                # remember that source_views_indexes is in ascending order
                terminal_logger.warning('One or more source views from the ones requested is out of bound.\n'
                                        f'\tIndexes requested: {source_views_indexes} -> '
                                        f'Ignoring frame of idx {frame_idx} from sequence {sequence_dir}.\n'
                                        f'\tThe sequence {sequence_dir} is of length {sequence_length}')
                continue

            # checks if any source source view is missing
            missing_source_view = False
            source_view_paths = self.get_source_view_paths_from_img_path_and_indexes(img_path, source_views_indexes)
            for source_view_index, source_view_img_path in zip(source_views_indexes, source_view_paths):
                if not source_view_img_path.exists():
                    terminal_logger.warning(f'Frame of idx {frame_idx} from sequence {sequence_dir} have the '
                                            f'source view of idx {source_view_index} missing from the ones requested.\n'
                                            f'\tIndexes requested: {source_views_indexes} -> '
                                            f'Ignoring frame of idx {frame_idx} from sequence {sequence_dir}.')
                    missing_source_view = True
                    # not breaking the loop so that we know all the source views that are unavailable

            if not missing_source_view:
                img_paths_and_source_views.append((img_path, source_view_paths))

        terminal_logger.info('Image list curated and source-views registered.')
        terminal_logger.info(f'Original list: {len(img_paths)} files')
        terminal_logger.info(f'Curated list: {len(img_paths_and_source_views)} files')

        return img_paths_and_source_views

    def get_source_view_paths_from_img_path_and_indexes(self, img_path, source_views_indexes):
        """
        Parameters
        ----------
        img_path: str
            Absolute or relative path to the data sample in KITTI Raw format.
            e.g.: path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
        source_views_indexes : list of int
            The relative indexes to sample from the neighbouring views of the target view.
            It is expected that the list is in ascending order and does not contains 0 (corresponding to the target).
            For example, source_indexes=[-1,1] will load the views at time t-1, t, t+1

        Returns
        -------
        int
            Absolute or relative path of the requested source view w.r.t to img_path

        Example
        -------
        img_path = "path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png"
        source_views_indexes = [-1,1]
        get_source_view_paths_from_indexes_and_img_path(img_path,source_views_indexes)
        >> ["path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000084.png",
        >> "path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000086.png"]
        """

        img_path = Path(img_path)
        frame_idx = self.get_frame_idx_from_image_path(img_path)
        source_views_img_paths = []
        for source_view_index in source_views_indexes:
            source_view_frame_idx = frame_idx + source_view_index
            source_view_img_path = img_path.parent / f'{source_view_frame_idx:010d}.png'
            source_views_img_paths.append(source_view_img_path)

        return source_views_img_paths

    def get_frame_idx_from_image_path(self, img_path):
        """
        Parameters
        ----------
        img_path: str
            Absolute or relative path to the data sample in KITTI Raw format.
            e.g.: path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png

        Returns
        -------
        int
            Index of the image path corresponding's frame
        """

        # e.g., img_path == path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
        img_path = Path(img_path)
        file_name = img_path.stem  # e.g., 0000000085.png
        frame_idx = int(file_name)  # 85

        return frame_idx


    def read_calib_file(self, calib_file_path):
        """
        In the calib file `calib_cam_to_cam.txt` the first two lines are:

        1. the data and time at which the calibration has been made
        2. corner distance

        The following lines are intrinsics/extrinsics/projection matrixes

        all lines are in the format:
        <matrix or data name>: <data>

        example (first three lines):
            calib_time: 09-Jan-2012 13:57:47
            corner_dist: 9.950000e-02
            S_00: 1.392000e+03 5.120000e+02


        we create a dict by splitting the line strings at the first colon character
        while handeling the `calib_time` case where multiple colon characters can be
        found in the same line
        """

        with calib_file_path.open() as calib_file:
            lines = calib_file.readlines()
            calibs = {l.split(":")[0]: ':'.join(l.split(":")[1:]) for l in lines}

        return calibs

    def get_intrinsics(self, calib_file_path):
        """
        In the calib file `calib_cam_to_cam.txt` the intrinsics of a camera can be extracted from
        its projective matrix (3x4 matrix)

        Parameters
        ----------
        calib_file_path : str
            path to the relevant calibration file

        Returns
        -------
        np array
            contains the camera intrinsics:
                fx  0  cx
                0  fy  cy
                0   0   1

        """
        calibs = self.read_calib_file(calib_file_path)
        K = np.array(calibs[INTRINSICS_MATRIX].split()).astype(float).reshape((3, 4))

        return K[:, :3]

    def get_intrinsics_for_all_sequences(self, split_file_path):
        """
        Returns a dictionary giving the intrinsics for each sequences in split_file.

        In truth, there is one calib file for each capture day, so all sequences under a same capture day have
        the same intrinsics

        Parameters
        ----------
        split_file_path: str
            The path to a .txt file containing the list the relative path to each image w.r.t. root dir.

        Returns
        -------
        dict
            A dictionary where the key is the capture date of a sequence and the value the corresponding intrinsics
        """

        # Under the root dir, KITTI raw is organized as follow
        # {capture_date}/{capture_date}_drive_{sequence_idx:04d}_sync/image_02/data/{frame_idx:010d}.png
        # e.g., 2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
        # thus, each line of the split file is assumed to have the same formatting
        arr = np.genfromtxt(split_file_path, delimiter='/', dtype=str)
        # each row of arr is of the form
        # ['2011_09_26', '2011_09_26_drive_0048_sync', 'image_02', 'data', '0000000085.png']
        used_capture_dates = np.unique(arr[:, 0])

        intrinsics = {}
        for capture_date in used_capture_dates:
            calib_file_path = self.kitti_root_dir / capture_date / CAMERA_CALIBRATION_FILE_NAME
            assert calib_file_path.exists(), calib_file_path
            intrinsics[capture_date] = self.get_intrinsics(calib_file_path)

        return intrinsics

    def read_npz_depth(self, file):
        """Reads a .npz depth map from https://github.com/TRI-ML/packnet-sfm/."""
        depth = np.load(file)['velodyne_depth'].astype(np.float32)
        return np.expand_dims(depth, axis=2)

    def read_png_depth(self, file):
        """Reads a .png depth map."""
        depth_png = np.array(Image.open(file), dtype=int)
        assert (np.max(depth_png) > 255), 'Wrong .png depth file'
        depth = depth_png.astype(np.float) / 256.
        depth[depth_png == 0] = -1.
        return np.expand_dims(depth, axis=2)

    def __getitem__(self, idx):
        img_path, source_views_paths = self.samples_and_source_views_paths[idx]

        img = Image.open(img_path)
        if self.input_channels == 'gray':
            img = img.convert('L')

        sample = {'target_view': img, 'idx': idx}

        if self.source_views_requested:
            source_views_imgs = [Image.open(path) for path in source_views_paths]
            if self.input_channels == 'gray':
                source_views_imgs = [source_views_img.convert('L') for source_views_img in source_views_imgs]
            sample['source_views'] = source_views_imgs

        # e.g., img_path == path_to_dataset_root_dir/2011_09_26/2011_09_26_drive_0048_sync/image_02/data/0000000085.png
        capture_date = Path(img_path).parents[3].name # 2011_09_26
        K = self.intrinsics[capture_date]
        sample['intrinsics'] = K

        sequence_idx = Path(img_path).parents[2].name[17:21]  # 0048
        frame_idx = Path(img_path).stem
        sample['filename'] = f"{capture_date}_{sequence_idx}_{frame_idx}"

        if 'val' in self.split_name or 'test' in self.split_name:

            path_suffix = f"{capture_date}_drive_{sequence_idx}_sync/{PROJECTED_GROUNDTRUTH_DIR}/{frame_idx}.png"

            projected_lidar_path = Path(self.gt_depth_root_dir) / 'val' / path_suffix

            if not projected_lidar_path.exists:
                projected_lidar_path = Path(self.gt_depth_root_dir) / 'train' / path_suffix

            projected_lidar = self.read_png_depth(projected_lidar_path)
            sample['projected_lidar'] = projected_lidar

        if self.load_sparse_depth:
            # assumes the depth files are stored in the same format as KITTI_raw:
            # depth_root_dir/2011_09_26/2011_09_26_drive_0048_sync/proj_depth/velodyne/image_02/0000000085.npz
            depth_path = Path(self.sparse_depth_root_dir) / capture_date \
                         / f"{capture_date}_drive_{sequence_idx}_sync" / PROJECTED_VELODYNE_DIR / f"{frame_idx}.npz"

            depth = self.read_npz_depth(str(depth_path))
            sample['sparse_projected_lidar'] =  depth


        if self.data_transform is not None:
            sample = self.data_transform(sample, **self.data_transform_options)


        return sample