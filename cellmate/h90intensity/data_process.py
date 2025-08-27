import numpy as np
import pandas as pd
from skimage import morphology
import scipy.ndimage as ndi
from tqdm import trange
from scipy.ndimage import gaussian_filter1d

# def normalize_fluorescence(F, B,
#                            method="dff",      # "dff" or "snr"
#                            baseline="auto",   # float, or "auto"
#                            roll_seconds=60,   # for auto baseline
#                            fps=None,          # needed if baseline="auto" and roll_seconds given
#                            low_pct=20,        # rolling percentile for baseline
#                            bleach=None,       # None, "exp", or "lin"
#                            eps=1e-9):
#     """
#     F: 1D array of fluorescence (cell ROI)
#     B: 1D array of background (bg ROI)
#     area_*: ROI pixel counts to scale background
#     """
#     F = np.asarray(F, float)
#     B = np.asarray(B, float)
#     assert F.shape == B.shape, "F and B must have same length"

#     # per-frame background subtraction
#     Fcorr = F - B

#     # optional bleach correction
#     if bleach is not None:
#         t = np.arange(len(Fcorr), dtype=float)
#         if bleach == "lin":
#             # linear trend
#             A = np.vstack([t, np.ones_like(t)]).T
#             m, c = np.linalg.lstsq(A, Fcorr, rcond=None)[0]
#             trend = m*t + c
#             trend = np.clip(trend, eps, None)
#             Fcorr = Fcorr / trend * np.median(trend)
#         elif bleach == "exp":
#             # log-linear fit for exponential bleaching: F ~ a*exp(b t)
#             y = np.log(np.clip(Fcorr - np.min(Fcorr) + eps, eps, None))
#             A = np.vstack([t, np.ones_like(t)]).T
#             b, loga = np.linalg.lstsq(A, y, rcond=None)[0]
#             trend = np.exp(loga) * np.exp(b*t)
#             trend = np.clip(trend, eps, None)
#             Fcorr = Fcorr / trend * np.median(trend)

#     # baseline
#     if isinstance(baseline, (int, float)):
#         F0 = float(baseline)
#     else:
#         if fps is None:
#             # fallback: global low percentile if fps unknown
#             F0 = np.percentile(Fcorr, low_pct)
#         else:
#             win = max(3, int(round(roll_seconds * fps)))
#             s = pd.Series(Fcorr)
#             F0_series = s.rolling(win, center=True, min_periods=1)\
#                          .apply(lambda v: np.percentile(v, low_pct), raw=True)
#             # choose a single F0 (median of rolling baseline) for classic ΔF/F0
#             F0 = float(np.median(F0_series.values))

#     F0 = max(F0, eps)

#     if method.lower() == "dff":
#         out = (Fcorr - F0) / F0
#     elif method.lower() == "snr":
#         out = Fcorr / np.maximum(B, eps)
#     else:
#         raise ValueError("method must be 'dff' or 'snr'")

#     return out, Fcorr, F0


# def smooth_trace(trace, window=5, method="linear"):
#     trace = np.asarray(trace, float)

#     # replace zeros with NaN
#     trace[trace == 0] = np.nan

#     # interpolate missing values
#     s = pd.Series(trace)
#     trace_interp = s.interpolate(method=method, limit_direction="both").to_numpy()

#     # apply moving average smoothing
#     smoothed = pd.Series(trace_interp).rolling(window, center=True, min_periods=1).mean().to_numpy()

#     return smoothed, trace_interp


# def post_process(F, B):
#     smoothed, filled = smooth_trace(F, window=3)
#     dff, Fcorr, F0 = normalize_fluorescence(
#         smoothed, B,
#         area_cell=1, area_bg=1,
#         method="dff",
#         baseline="auto", roll_seconds=60, fps=5,
#         low_pct=20,
#         bleach=None   # try None, "lin", or "exp"
#     )
#     return Fcorr

def intensity_h90_diff_local_single(fluorescent_image, mask, kernel, erosion_k=11, threshold=90, ):
    closed_image = morphology.binary_erosion(mask, footprint=morphology.disk(erosion_k))
    response = ndi.convolve((fluorescent_image.astype(np.float_)), kernel, mode='constant', cval=0)
    response[~closed_image] = 0

    idx = np.argmax(response)
    row, col = np.unravel_index(idx, response.shape)

    response_list = response[response > 0]
    thresh = 0
    if len(response_list > 0):
        thresh = np.percentile(response_list, threshold)
    # Generate binary image by threshold
    binary = response > thresh
    nc_mask = post_process(binary, area_threshold=200)

    nc_flag = nc_mask[row, col]

    mask_edge = ~closed_image & mask

    edge_list = fluorescent_image[mask_edge]
    edge_intensity = np.mean(edge_list[edge_list > np.percentile(edge_list, threshold)])

    if nc_flag:
        core_intensity = np.mean(fluorescent_image[nc_mask])
        cy_intensity = np.mean(fluorescent_image[~nc_mask & closed_image])
    else:
        core_intensity = 0
        cy_intensity = np.mean(fluorescent_image[closed_image])


    if nc_flag & (core_intensity<cy_intensity*1.2):
        nc_flag = False
        core_intensity = 0

    return nc_flag, core_intensity, edge_intensity, cy_intensity


def dot_detector_kernel(radius=10):
    """Kernel to detect bright round dots of given radius"""
    # size = radius*2 + 1
    
    # circular mask for the dot
    mask = morphology.disk(radius).astype(float)
    
    # normalize: positive inside, negative outside
    kernel = mask - mask.mean()
    
    # normalize to unit L1 norm (optional)
    kernel /= np.sum(np.abs(kernel))
    return kernel


def post_process(image, selem=morphology.disk(3), area_threshold=100, erosion_factor=2):
    """
    Apply morphological post-processing to clean up a binary image.

    Steps:
    1. Opening to remove small white noise.
    2. Closing to fill small black holes.
    3. Remove small objects below a specified area threshold.
    4. Erode the regions to shrink them slightly and reduce noise.

    Parameters:
    ----------
    image : 2D binary numpy array
        Input binary mask to be cleaned.
    selem : ndarray, optional
        Structuring element used for morphological operations (default: disk(3)).
    area_threshold : int, optional
        Minimum area of objects to keep (default: 100 pixels).
    erosion_factor : int, optional
        Radius of erosion to apply after cleaning (default: 2).

    Returns:
    -------
    shrunk : 2D binary numpy array
        Cleaned and eroded binary mask.
    """

    # Morphological opening: remove small white noise
    opened = morphology.opening(image, selem)

    # Morphological closing: fill small black holes
    closed = morphology.closing(opened, selem)

    # Remove small connected components
    img_rm_small = morphology.remove_small_objects(closed, min_size=area_threshold, connectivity=2)
    return img_rm_small


def instance_fluorescent_intensity_h90(fluorescent_image, masks, kernel, *args, **kwargs):
    data_sheet = []
    label_list = np.unique(masks)[1:]
    bg_intensity = np.mean(fluorescent_image[masks == 0])
    for label in label_list:
        mask = masks == label
        x, y = np.where(mask)
        x_min = max(0, np.min(x) - 10)
        x_max = min(np.max(x)+10, mask.shape[0])
        y_min = max(0, np.min(y) - 10)
        y_max = min(np.max(y)+10, mask.shape[1])

        mask = mask[x_min:x_max, y_min:y_max]
        f_cropped = fluorescent_image[x_min:x_max, y_min:y_max]
        nc_flag, core_intensity, edge_intensity, cy_intensity = intensity_h90_diff_local_single(f_cropped,
                                                                                                mask,
                                                                                                kernel=kernel,
                                                                                                *args, **kwargs)
        data = [label, nc_flag, core_intensity, edge_intensity, cy_intensity, bg_intensity]
        data_sheet.append(data)
    return data_sheet


def get_intensity_table(fluorescent_image, tracked_image, kernel=None, *args, **kwargs):
    data_sheet = None
    if kernel is None:
        kernel = dot_detector_kernel(radius=19)
    for frame_number in trange(0, tracked_image.shape[0]):
        data = instance_fluorescent_intensity_h90(fluorescent_image[frame_number],
                                                  tracked_image[frame_number],
                                                  kernel,
                                                  *args, **kwargs)
        data = pd.DataFrame(data, columns=["label",
                                           "nc_flag",
                                           "nuclear",
                                           "membrane",
                                           "cytoplasmic",
                                           "background"])
        data["frame"] = frame_number
        data_sheet = pd.concat([data_sheet, data])
    return data_sheet


def smooth_normalize(data_sheet):
    data_sheet["norm_nuclear"] = (data_sheet["nuclear"] - data_sheet["background"]).clip(lower=0)#/ merged_data_test["background"]
    data_sheet["norm_membrane"] = (data_sheet["membrane"] - data_sheet["background"]).clip(lower=0) #/ merged_data_test["background"]
    data_sheet["norm_cytoplasmic"] = (data_sheet["cytoplasmic"] - data_sheet["background"]).clip(lower=0) #/ merged_data_test["background"]

    all_label = data_sheet.label.unique()
    for i in range(0, len(all_label)):
        current_label = all_label[i]
        current_mask = data_sheet.label==current_label
        data_i = data_sheet[current_mask]

        data_sheet.loc[current_mask, "norm_nuclear"] = gaussian_filter1d(data_i["norm_nuclear"], sigma=1, mode="nearest")
        data_sheet.loc[current_mask, "norm_membrane"] = gaussian_filter1d(data_i["norm_membrane"], sigma=1, mode="nearest")
        data_sheet.loc[current_mask, "norm_cytoplasmic"] = gaussian_filter1d(data_i["norm_cytoplasmic"], sigma=1, mode="nearest")

    filter_mask = data_sheet["norm_nuclear"] < (data_sheet["norm_cytoplasmic"]*1.5)
    data_sheet.loc[filter_mask, "norm_nuclear"] = 0
    data_sheet.loc[filter_mask, "nc_flag"] = False

    for i in range(0, len(all_label)):
        current_label = all_label[i]
        current_mask = data_sheet.label==current_label
        data_i = data_sheet[current_mask]

        smoothed = data_i.norm_nuclear.rolling(window=20, center=True, min_periods=5).mean()
        smoothed_diff = smoothed.diff()
        slope_nc = smoothed_diff[smoothed_diff > 0].sum()

        smoothed = data_i.norm_membrane.rolling(window=20, center=True, min_periods=5).mean()
        smoothed_diff = smoothed.diff()
        slope_mb = smoothed_diff[smoothed_diff > 0].sum()

        data_sheet.loc[current_mask, "slope_nc"] = slope_nc
        data_sheet.loc[current_mask, "slope_mb"] = slope_mb
        data_sheet.loc[current_mask, "slope_marker"] = slope_mb >= slope_nc
    return data_sheet
