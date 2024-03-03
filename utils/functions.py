"""
Useful Functions.

Authors: Hongjie Fang.
"""
import numpy as np
import mindspore.numpy as mnp
import mindspore.ops as ops
from mindspore import Tensor
import mindspore.nn as nn
from mindspore import context
from mindspore.ops import operations as P

def display_results(metrics_dict, logger):
    """
    Given a metrics dict, display the results using the logger.

    Parameters
    ----------
        
    metrics_dict: dict, required, the given metrics dict;
    logger: logging.Logger object, the logger.
    """
    try:
        display_list = []
        for key in metrics_dict.keys():
            if key == 'samples':
                num_samples = metrics_dict[key]
            else:
                display_list.append([key, float(metrics_dict[key])])
        logger.info("Metrics on {} samples:".format(num_samples))
        for display_line in display_list:
            metric_name, metric_value = display_line
            logger.info("  {}: {:.6f}".format(metric_name, metric_value))    
    except Exception:
        logger.warning("Unable to display the results, the operation is ignored.")
        pass


def gradient(x):
    """
    Get gradient of xyz image for MindSpore.

    Parameters
    ----------
    
    x: Tensor, the xyz map to get gradient.

    Returns
    -------
    
    The x-axis-in-image gradient and y-axis-in-image gradient of the xyz map.
    """
    left = x
    right = ops.Pad(((0, 0), (0, 0), (0, 0), (0, 1)))(x)[:, :, :, 1:]
    top = x
    bottom = ops.Pad(((0, 0), (0, 0), (0, 1), (0, 0)))(x)[:, :, 1:, :]

    dx, dy = right - left, bottom - top
    dx[:, :, :, -1] = mnp.zeros_like(dx[:, :, :, -1])
    dy[:, :, -1, :] = mnp.zeros_like(dy[:, :, -1, :])
    return dx, dy

def get_surface_normal_from_xyz(x, epsilon=1e-8):
    """
    [MindSpore Version] Get the surface normal of xyz image.

    Parameters
    ----------
    x: the xyz map to get surface normal;
    epsilon: float, optional, default: 1e-8, the epsilon to avoid nan.

    Returns
    -------
    The surface normals.
    """
    dx, dy = gradient(x)
    surface_normal = P.Cross()(dx, dy)
    norm = P.Sqrt()(P.ReduceSum(keep_dims=True)(P.Square()(surface_normal), axis=1))
    surface_normal = surface_normal / (norm + epsilon)
    return surface_normal

def get_xyz(depth, fx, fy, cx, cy, original_size=(1280, 720)):
    """
    Get XYZ from depth image and camera intrinsics for MindSpore.

    Parameters
    ----------
    depth: Tensor, the depth image;
    fx, fy, cx, cy: Tensor, the camera intrinsics;
    original_size: tuple of (int, int), optional, default: (1280, 720), the original size of the image.

    Returns
    -------
    The XYZ value of each pixel.
    """
    bs, h, w = depth.shape
    indices = np.indices((h, w), dtype=np.float32)
    indices = Tensor(np.array([indices] * bs), dtype=mnp.float32)
    x_scale = w / original_size[0]
    y_scale = h / original_size[1]
    fx = fx * x_scale
    fy = fy * y_scale
    cx = cx * x_scale
    cy = cy * y_scale
    z = depth
    tile_op = P.Tile()

    tiles = (1, h, w)

    x = (indices[:, 1, :, :] - tile_op(fx, tiles)) * z / tile_op(fy, tiles)
    y = (indices[:, 0, :, :] - tile_op(cx, tiles)) * z / tile_op(cy, tiles)
    return mnp.stack([x, y, z], axis=1)


def get_surface_normal_from_depth(depth, fx, fy, cx, cy, original_size=(1280, 720), epsilon=1e-8):
    """
    Get surface normal from depth and camera intrinsics for MindSpore.

    Parameters
    ----------
    depth: Tensor, the depth image;
    fx, fy, cx, cy: Tensor, the camera intrinsics;
    original_size: tuple, the original size of image;
    epsilon: float, small value to avoid division by zero.

    Returns
    -------
    The surface normals.
    """
    xyz = get_xyz(depth, fx, fy, cx, cy, original_size=original_size)
    return get_surface_normal_from_xyz(xyz, epsilon=epsilon)


def to_device(data_dict, device_type):
    """
    (Simulated) Put the data in the data_dict to the specified device for MindSpore.
    
    Parameters
    ----------
    data_dict: dict, contains tensors;
    device_type: str, type of the device ('CPU', 'GPU', etc.).

    Returns
    -------
    The final data_dict.
    """
    context.set_context(device_type=device_type)
    # In practice, MindSpore handles device placement automatically,
    # so you might not need to manually move each tensor.
    return data_dict