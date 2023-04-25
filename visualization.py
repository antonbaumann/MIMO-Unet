import matplotlib.pyplot as plt
import matplotlib.cm
import matplotlib.colors
import numpy as np
from typing import Tuple

import torch

import ndvi_prediction.utils
from pathlib import Path


def plot_ndvi_correlation(ndvi_true: np.ndarray, ndvi_predicted: np.ndarray):
    """Plot a correlation between predicted and true ndvi for an image.
    Idea from Zhao2020: Deeply synergistic optical and SAR time series for crop dynamic monitoring"""
    ndvi_true = ndvi_true.flatten()
    ndvi_predicted = ndvi_predicted.flatten()
    assert ndvi_true.shape == ndvi_predicted.shape
    # data = np.vstack((ndvi_true, ndvi_predicted))
    corr_matrix = np.corrcoef(ndvi_true, ndvi_predicted)
    r2 = corr_matrix[0, 1] ** 2
    rmse = np.sqrt(np.mean((ndvi_predicted - ndvi_true) ** 2))  #

    # change colormap do exclude empty places
    orig_cmap = plt.get_cmap("viridis", 256)
    upper = np.array([orig_cmap(i) for i in np.arange(0, 1, 0.01)])
    z = [0, 0, 0, 0]
    cmap = np.vstack((z, upper))
    cmap = matplotlib.colors.ListedColormap(cmap, name="myColorMap", N=cmap.shape[0])
    plt.figure()
    plt.hist2d(ndvi_true, ndvi_predicted, bins=100, cmap=cmap)
    plt.xlim(0, 1)
    plt.title(f"R²={r2:.4f}  RMSE={rmse:.4f}")
    # plt.savefig("/tmp/test.png")
    plt.show()


def data_debug_figure(
    vv: np.ndarray = None,
    vh: np.ndarray = None,
    cr: np.ndarray = None,
    nrpb: np.ndarray = None,
    lc: np.ndarray = None,
    ndvi: np.ndarray = None,
    ndvi_predicted: np.ndarray = None,
    rgb: np.ndarray = None,
    dpi: int = 150,
    ndvi_range=(-1, 1),
    ndvi_cmap: str = "RdYlGn",
):
    """Saves a figure with the image data."""
    bins = 50
    cols = sum(
        [int(ar is not None) for ar in [vv, cr, nrpb, lc, ndvi, ndvi_predicted, rgb]]
    )
    assert bool(vv is not None) + bool(vv is not None) in {
        0,
        2,
    }, "Provide either VV AND VH or none of both!"
    fig, axes = plt.subplots(nrows=2, ncols=cols)
    fig: plt.Figure
    fig.dpi = dpi
    fig.set_size_inches(w=8, h=3)
    axes: Tuple[plt.Axes]
    list(map(lambda ax: ax.axis("off"), axes[0]))
    list(map(lambda ax: ax.get_yaxis().set_visible(False), axes[1]))
    # for ar in [vv, vh, cr, nrpb, lc, ndvi]:
    #     if ar is not None:
    #         assert ar.shape == (256, 256), f"shape wrong: {ar.shape}!=(256,256)"
    # if rgb is not None:
    #     assert rgb.shape == (256, 256, 3), f"rgb shape wrong: {rgb.shape}!=(256,256,3)"
    col = 0
    if rgb is not None:
        rgb_enhanced = ndvi_prediction.utils.clip_extremes(rgb)
        axes[0, col].imshow(ndvi_prediction.utils.contrast_stretch(rgb_enhanced))
        axes[0, col].set_title("RGB")
        axes[1, col].hist(rgb[..., 0].flatten(), alpha=0.4, color="red", bins=bins)
        axes[1, col].hist(rgb[..., 1].flatten(), alpha=0.4, color="green", bins=bins)
        axes[1, col].hist(rgb[..., 2].flatten(), alpha=0.4, color="blue", bins=bins)
        col += 1
    if vv is not None and vh is not None:
        sar = ndvi_prediction.utils.get_sar_greenblue(
            vv=ndvi_prediction.utils.contrast_stretch(vv), vh=ndvi_prediction.utils.contrast_stretch(vh)
        )
        axes[0, col].imshow(sar)
        axes[0, col].set_title("Backscatter")
        axes[1, col].hist(vv.flatten(), bins=bins, alpha=0.5, color="green")
        axes[1, col].hist(vh.flatten(), bins=bins, alpha=0.5, color="blue")
        col += 1
    if cr is not None:
        _cr = ndvi_prediction.utils.clip_extremes(cr, percentile=1)
        _img = axes[0, col].imshow(ndvi_prediction.utils.contrast_stretch(_cr))
        axes[0, col].set_title("CR")
        fig.colorbar(_img, ax=axes[0, col])
        axes[1, col].hist(_cr.flatten(), bins=bins)
        col += 1
    if nrpb is not None:
        _img = axes[0, col].imshow(
            ndvi_prediction.utils.contrast_stretch(nrpb), vmin=-1, vmax=1, cmap="Spectral"
        )
        fig.colorbar(_img, ax=axes[0, col])

        axes[0, col].set_title("NRPB")
        axes[1, col].hist(nrpb.flatten(), bins=bins)
        col += 1
    if lc is not None:
        axes[0, col].imshow(ndvi_prediction.utils.contrast_stretch(lc))
        axes[0, col].set_title("Landcover")
        axes[1, col].hist(lc.flatten(), bins=bins)
        col += 1
    if ndvi_predicted is not None:
        _img = axes[0, col].imshow(
            ndvi_predicted, cmap=ndvi_cmap, vmin=ndvi_range[0], vmax=ndvi_range[1]
        )
        fig.colorbar(_img, ax=axes[0, col])
        axes[0, col].set_title("NDVI pred")
        ndvi_pred_hist = axes[1, col].hist(ndvi_predicted.flatten(), bins=bins)
        col += 1
    if ndvi is not None:
        _img = axes[0, col].imshow(
            ndvi, cmap=ndvi_cmap, vmin=ndvi_range[0], vmax=ndvi_range[1]
        )
        fig.colorbar(_img, ax=axes[0, col])
        axes[0, col].set_title("NDVI")
        ndvi_hist = axes[1, col].hist(ndvi.flatten(), bins=bins)
    if ndvi is not None and ndvi_predicted is not None:
        # set the xaxis to the same range
        x_min_bins = min(np.min(ndvi_pred_hist[1]), np.min(ndvi_hist[1]))
        x_max_bins = max(np.max(ndvi_pred_hist[1]), np.max(ndvi_hist[1]))
        axes[1, col].set_xlim(left=x_min_bins, right=x_max_bins)
        axes[1, col - 1].set_xlim(left=x_min_bins, right=x_max_bins)
    fig.tight_layout()
    return fig, axes


# Original from
# https://gist.github.com/jimfleming/c1adfdb0f526465c99409cc143dea97b#gistcomment-2398882
# and then adopted
import matplotlib


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


def save_figure(fig: plt.Figure, path: str) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(p)


if __name__ == "__main__":
    from numpy.random import multivariate_normal
    from numpy import random

    ndvi_true = random.random((256, 256))
    ndvi_pred = random.random((256, 256))
    ndvi_true = np.clip(ndvi_true, 0.2, 1)
    plot_ndvi_correlation(ndvi_true, ndvi_pred)
