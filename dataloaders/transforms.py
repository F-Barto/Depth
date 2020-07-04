from functools import partial
import cv2
import numpy as np
import random
import torchvision.transforms as transforms
from PIL import Image

########################################################################################################################

TARGET_VIEW = 'target_view'
SOURCE_VIEWS = 'source_views'
GT_DEPTH = 'projected_lidar'
SPARSE_DEPTH = 'sparse_projected_lidar'
DEPTH_KEYS = [GT_DEPTH, SPARSE_DEPTH]

def resize_image(image, shape, interpolation=Image.ANTIALIAS):
    """
    Resizes input image.
    Parameters
    ----------
    image : Image.PIL
        Input image
    shape : tuple [H,W]
        Output shape
    interpolation : int
        Interpolation mode
    Returns
    -------
    image : Image.PIL
        Resized image
    """
    transform = transforms.Resize(shape, interpolation=interpolation)
    return transform(image)

def resize_depth(depth, shape):
    """
    Resizes depth map.
    Parameters
    ----------
    depth : np.array [h,w]
        Depth map
    shape : tuple (H,W)
        Output shape
    Returns
    -------
    depth : np.array [H,W]
        Resized depth map
    """
    depth = cv2.resize(depth, dsize=shape[::-1],
                       interpolation=cv2.INTER_NEAREST)
    return depth


def resize_sample_image_and_intrinsics(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes the image and intrinsics of a sample
    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode
    Returns
    -------
    sample : dict
        Resized sample
    """
    # Resize image and corresponding intrinsics
    image_transform = transforms.Resize(shape, interpolation=image_interpolation)
    (orig_w, orig_h) = sample[TARGET_VIEW].size
    (out_h, out_w) = shape

    # Scale intrinsics
    key = 'intrinsics'
    intrinsics = np.copy(sample[key])
    intrinsics[0] *= out_w / orig_w
    intrinsics[1] *= out_h / orig_h
    sample[key] = intrinsics

    # Scale target image
    key = TARGET_VIEW
    sample[key] = image_transform(sample[key])

    # Scale source views images
    key = SOURCE_VIEWS
    sample[key] = [image_transform(source_view) for source_view in sample[key]]


    # Return resized sample
    return sample

def resize_sample(sample, shape, image_interpolation=Image.ANTIALIAS):
    """
    Resizes a sample, including image, intrinsics and depth maps.
    Parameters
    ----------
    sample : dict
        Dictionary with sample values
    shape : tuple (H,W)
        Output shape
    image_interpolation : int
        Interpolation mode
    Returns
    -------
    sample : dict
        Resized sample
    """

    # Resize image and intrinsics
    sample = resize_sample_image_and_intrinsics(sample, shape, image_interpolation)

    # Resize depth maps
    for key in DEPTH_KEYS:
        if sample.get(key) is not None:
            sample[key] = resize_depth(sample[key], shape)

    # Resize depth contexts (not used currently)
    key = 'depth_source_views'
    if sample.get(key) is not None:
        sample[key] = [resize_depth(k, shape) for k in sample[key]]

    # Return resized sample
    return sample


def to_tensor(image, tensor_type='torch.FloatTensor'):
    """Casts an image to a torch.Tensor"""
    transform = transforms.ToTensor()
    return transform(image).type(tensor_type)

def to_tensor_sample(sample, tensor_type='torch.FloatTensor'):
    """
    Casts the keys of sample to tensors.
    Parameters
    ----------
    sample : dict
        Input sample
    tensor_type : str
        Type of tensor we are casting to
    Returns
    -------
    sample : dict
        Sample with keys cast as tensors
    """

    # Convert single items
    for key in ([TARGET_VIEW, TARGET_VIEW + '_original'] + DEPTH_KEYS):
        if sample.get(key) is not None:
            sample[key] = to_tensor(sample[key], tensor_type)

    # Convert lists
    for key in [SOURCE_VIEWS, SOURCE_VIEWS + '_original']:
        if sample.get(key) is not None:
            sample[key] = [to_tensor(source_view, tensor_type) for source_view in sample[key]]

    # Return converted sample
    return sample


def duplicate_sample(sample):
    """
    Duplicates sample images and contexts to preserve their unaugmented versions.
    Parameters
    ----------
    sample : dict
        Input sample
    Returns
    -------
    sample : dict
        Sample including [+"_original"] keys with copies of images and contexts.
    """

    # Duplicate target image
    key = TARGET_VIEW
    sample[f'{key}_original'] = sample[key].copy()

    # Duplicate source view images
    key = SOURCE_VIEWS
    sample[f'{key}_original'] = [k.copy() for k in sample[key]]

    # Return duplicated sample
    return sample

def colorjitter_sample(sample, parameters, prob=1.0):
    """
    Jitters input images as data augmentation.
    Parameters
    ----------
    sample : dict
        Input sample
    parameters : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    prob : float
        Jittering probability
    Returns
    -------
    sample : dict
        Jittered sample
    """
    if random.random() < prob:
        # Prepare transformation
        color_augmentation = transforms.ColorJitter()
        brightness, contrast, saturation, hue = parameters
        augment_image = color_augmentation.get_params(
            brightness=[max(0, 1 - brightness), 1 + brightness],
            contrast=[max(0, 1 - contrast), 1 + contrast],
            saturation=[max(0, 1 - saturation), 1 + saturation],
            hue=[-hue, hue])

        # Jitter target image
        key = TARGET_VIEW
        sample[key] = augment_image(sample[key])

        # Jitter source views images
        key = SOURCE_VIEWS
        sample[key] = [augment_image(source_view) for source_view in sample[key]]

    # Return jittered (?) sample
    return sample

###################################################

def train_transforms(sample, image_shape, jittering=None):
    """
    Training data augmentation transformations
    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    jittering : tuple (brightness, contrast, saturation, hue)
        Color jittering parameters
    Returns
    -------
    sample : dict
        Augmented sample
    """
    image_shape = tuple(image_shape)
    sample = resize_sample(sample, image_shape)
    sample = duplicate_sample(sample)
    if jittering is not None and len(jittering) > 0:
        jittering = tuple(jittering)
        sample = colorjitter_sample(sample, jittering)
    sample = to_tensor_sample(sample)
    return sample

def val_transforms(sample, image_shape):
    """
    Validation data augmentation transformations
    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    Returns
    -------
    sample : dict
        Augmented sample
    """
    image_shape = tuple(image_shape)

    sample[TARGET_VIEW] = resize_image(sample[TARGET_VIEW], image_shape)

    # Resize depth maps
    for key in DEPTH_KEYS:
        if sample.get(key) is not None:
            sample[key] = resize_depth(sample[key], image_shape)

    sample = to_tensor_sample(sample)
    return sample

def test_transforms(sample, image_shape):
    """
    Test data augmentation transformations
    Parameters
    ----------
    sample : dict
        Sample to be augmented
    image_shape : tuple (height, width)
        Image dimension to reshape
    Returns
    -------
    sample : dict
        Augmented sample
    """
    image_shape = tuple(image_shape)

    sample[TARGET_VIEW] = resize_image(sample[TARGET_VIEW], image_shape)

    # Resize depth maps
    for key in DEPTH_KEYS:
        if sample.get(key) is not None:
            sample[key] = resize_depth(sample[key], image_shape)

    sample = to_tensor_sample(sample)
    return sample