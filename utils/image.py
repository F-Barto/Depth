import torch.nn.functional as F
from utils.misc import same_shape
from numpy import unravel_index

def gradient_x(image):
    """
    Calculates the gradient of an image in the x dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image
    Returns
    -------
    gradient_x : torch.Tensor [B,3,H,W-1]
        Gradient of image with respect to x
    """
    return image[:, :, :, :-1] - image[:, :, :, 1:]

def gradient_y(image):
    """
    Calculates the gradient of an image in the y dimension
    Parameters
    ----------
    image : torch.Tensor [B,3,H,W]
        Input image
    Returns
    -------
    gradient_y : torch.Tensor [B,3,H-1,W]
        Gradient of image with respect to y
    """
    return image[:, :, :-1, :] - image[:, :, 1:, :]


def interpolate_image(image, shape, mode='nearest'):
    """
    Interpolate an image to a different resolution
    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Image to be interpolated
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation
    Returns
    -------
    image : torch.Tensor [B,?,H,W]
        Interpolated image
    """
    # Take last two dimensions as shape
    if len(shape) > 2:
        shape = shape[-2:]
    # If the shapes are the same, do nothing
    if same_shape(image.shape[-2:], shape):
        return image
    else:
        # Interpolate image to match the shape
        return F.interpolate(image, size=shape, mode=mode)

def interpolate_scales(images, shape=None, mode='nearest'):
    """
    Interpolate list of images to the same shape
    Parameters
    ----------
    images : list of torch.Tensor [B,?,?,?]
        Images to be interpolated, with different resolutions
    shape : tuple (H, W)
        Output shape
    mode : str
        Interpolation mode
        read more at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
    align_corners : bool, optional
        True if corners will be aligned after interpolation
        read more at https://pytorch.org/docs/stable/nn.functional.html#torch.nn.functional.interpolate
    Returns
    -------
    images : list of torch.Tensor [B,?,H,W]
        Interpolated images, with the same resolution
    """
    # If no shape is provided, interpolate to highest resolution
    if shape is None:
        # assumes that the first image is always the biggest
        shape = images[0].shape

    # if shape is [B,C,H,W], extract H,W
    if len(shape) > 2:
        shape = shape[-2:]

    return [F.interpolate(image, shape, mode=mode, align_corners=None) for image in images]

def match_scales(image, targets, num_scales, mode='nearest'):
    """
    Interpolate one image to produce a list of images with the same shape as targets
    Parameters
    ----------
    image : torch.Tensor [B,?,h,w]
        Input image
    targets : list of torch.Tensor [B,?,?,?]
        Tensors with the target resolutions
    num_scales : int
        Number of considered scales
    mode : str
        Interpolation mode
    align_corners : bool
        True if corners will be aligned after interpolation
    Returns
    -------
    images : list of torch.Tensor [B,?,?,?]
        List of images with the same resolutions as targets
    """
    # For all scales
    images = []
    image_shape = image.shape[-2:]
    for i in range(num_scales):
        target_shape = targets[i].shape
        # If image shape is equal to target shape
        if same_shape(image_shape, target_shape):
            images.append(image)
        else:
            # Otherwise, interpolate
            images.append(interpolate_image(
                image, target_shape, mode=mode))
    # Return scaled images
    return images
