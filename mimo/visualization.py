import matplotlib.cm
import numpy as np
import torch

# Original from
# https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b#gistcomment-2398882
# and then adapted

def colorize(value: torch.Tensor, vmin=None, vmax=None, cmap=None) -> np.ndarray:
    """
    A utility function for Torch/Numpy that maps a grayscale image to a matplotlib
    colormap for use with TensorBoard image summaries.
    By default it will normalize the input value to the range 0..1 before mapping
    to a grayscale colormap.
    Arguments:
      - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
        [height, width, 1].
      - vmin: the minimum value of the range used for normalization.
        (Default: value minimum)
      - vmax: the maximum value of the range used for normalization.
        (Default: value maximum)
      - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
        (Default: Matplotlib default colormap)

    Returns a 4D uint8 tensor of shape [height, width, 4].
    """
    # torchvision.utils.make_grid creates image with 3 duplicated RGB channels
    # for coloriziation only one channel is used, assuming that all channels have the same
    # information
    if value.ndim == 3 and value.size(0) == 3:
        value = value[0]
    assert value.ndim == 2
    assert value.shape[0] > 1
    assert value.shape[1] > 1
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:  # when vmin == vmax, return a array filled with zeros, use multiplication with 0.
        value = value * 0.0
    # squeeze last dim if it exists
    value = value.squeeze()
    value = value.detach().cpu().numpy()

    cmapper = matplotlib.cm.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # returns: (n x m x 4)
    value = value[..., 0:3]  # convert RGBA to RGB
    return value
