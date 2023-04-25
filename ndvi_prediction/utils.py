import sys
import numpy as np

epsilon = sys.float_info.epsilon


def contrast_stretch(img: np.ndarray, min=None, max=None):
    assert not np.isnan(img).any(), "Input should not contain NaN!"
    channel_axis = None
    if img.ndim == 3:
        if img.shape[2] <= 5:
            channel_axis = (0, 1)
        else:
            raise NotImplementedError(
                "For ndim=3, at most five channels allowed as third dimension."
            )
    _img = clip_extremes(img, percentile=1)
    if channel_axis and not (min or max):
        _min = _img.min(axis=channel_axis)
        _max = _img.max(axis=channel_axis)
    else:
        _min = _img.min() if min is None else min
        _max = _img.max() if max is None else max
    if np.min(_min) < np.max(_max):
        diff = _max - _min
        assert diff.all() != 0, ("Difference should not contain zero to avoid ZeroDivision", diff)
        _img = (_img - _min) / diff
        assert _img.min() > -epsilon, _img.min()
        assert _img.max() < 1 + epsilon, _img.max()
    else:
        print("Contrast stretch not applicable to data with only one value! SKIPPING!")
        _img = img
    return _img


def get_sar_greenblue(vv: np.ndarray, vh: np.ndarray) -> np.ndarray:
    assert vv.ndim == vh.ndim == 2, f"dimension not 2 but {vv.ndim} and {vh.ndim}"
    assert vv.shape == vh.shape, f"shape not equal: {vv.shape} != {vh.shape}"
    red = np.zeros(vv.shape)
    sar_img = np.stack((red, vv, vh), axis=2)
    return sar_img


def clip_extremes(arr: np.ndarray, percentile: int = 1) -> np.ndarray:
    assert not np.isnan(arr).any(), "Data should not contain NaN!"
    low_percentile = np.percentile(arr, percentile)
    high_percentile = np.percentile(arr, 100 - percentile)
    arr_clipped = np.clip(arr, a_min=low_percentile, a_max=high_percentile)
    assert not np.isnan(arr_clipped).any(), "Data should not contain NaN!"
    return arr_clipped


def normalized_difference(arr1: np.ndarray, arr2: np.ndarray) -> np.ndarray:
    # avoid signed integer overflow
    if np.issubdtype(arr1.dtype, np.integer):
        arr1 = arr1.astype(np.int32)
    if np.issubdtype(arr2.dtype, np.integer):
        arr2 = arr2.astype(np.int32)

    diff = np.subtract(arr1, arr2)
    summ = np.add(arr1, arr2)
    ndiff = np.divide(diff, summ)
    assert ndiff.shape == arr1.shape == arr2.shape

    error_msg = ""
    error_msg += (
        "Division by zero. Normalized difference cannot be calculated."
        if np.any(np.isnan(ndiff))
        else ""
    )
    error_msg += (
        "Error in normalized difference range!"
        if ndiff.min() <= (-1 - epsilon) or ndiff.max() >= (1 + epsilon)
        else ""
    )
    if error_msg:
        print(
            f"ndiff.dtype={ndiff.dtype}, arr1.dtype={arr1.dtype}, arr2.dtype={arr2.dtype}"
        )
        print(
            f"a1.min={arr1.min()}, a1.max={arr1.max()}, a2.min={arr2.min()}, a2.max={arr2.max()}"
        )
        print(f"ndvi: min={ndiff.min()} max={ndiff.max()}")
        raise ValueError(error_msg)
    return ndiff
