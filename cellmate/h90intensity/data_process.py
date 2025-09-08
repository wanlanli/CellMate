import numpy as np
import pandas as pd
from skimage import morphology, measure
import scipy.ndimage as ndi
from tqdm import trange
from scipy.ndimage import gaussian_filter1d


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

    if nc_flag & (core_intensity < cy_intensity*1.2):
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


def post_process(image, selem=morphology.disk(3), area_threshold=150, erosion_factor=2):
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

    # Erode the regions slightly to refine the boundaries
    # shrunk = erosion(img_rm_small, disk(erosion_factor))

    labeled = measure.label(img_rm_small)
    props = measure.regionprops(labeled)

    good_labels = 0
    good_label_ecc = 1
    for p in props:
        area = p.area
        if area < area_threshold:
            continue
        ecc = p.eccentricity
        if ecc > 0.75:           # too elongated
            continue
        if p.solidity < 0.85:      # concave / ring-like shapes
            continue
        if ecc < good_label_ecc:
            good_label_ecc = ecc
            good_labels = p.label
    # construct final mask
    if good_labels != 0:
        final_mask = labeled == good_labels
        # print("eff: ", good_label_ecc)
    else:
        final_mask = np.zeros(labeled.shape, dtype=bool)
    return final_mask


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
    data_sheet["norm_membrane"] = (data_sheet["membrane"] - data_sheet["background"] - 50).clip(lower=0) #/ merged_data_test["background"]
    data_sheet["norm_cytoplasmic"] = (data_sheet["cytoplasmic"] - data_sheet["background"]).clip(lower=0) #/ merged_data_test["background"]

    all_label = data_sheet.label.unique()
    for i in range(0, len(all_label)):
        current_label = all_label[i]
        current_mask = data_sheet.label==current_label
        data_i = data_sheet[current_mask]

        data_sheet.loc[current_mask, "norm_nuclear"] = gaussian_filter1d(data_i["norm_nuclear"], sigma=2, mode="nearest")
        data_sheet.loc[current_mask, "norm_membrane"] = gaussian_filter1d(data_i["norm_membrane"], sigma=2, mode="nearest")
        data_sheet.loc[current_mask, "norm_cytoplasmic"] = gaussian_filter1d(data_i["norm_cytoplasmic"], sigma=2, mode="nearest")

    filter_mask = data_sheet["norm_nuclear"] < (data_sheet["norm_cytoplasmic"]*1.5)
    data_sheet.loc[filter_mask, "norm_nuclear"] = 0
    data_sheet.loc[filter_mask, "nc_flag"] = False

    for i in range(0, len(all_label)):
        current_label = all_label[i]
        current_mask = data_sheet.label==current_label
        data_i = data_sheet[current_mask]

        smoothed = data_i.norm_nuclear.rolling(window=5, center=True, min_periods=5).mean()
        smoothed_diff = smoothed.diff()
        slope_nc = smoothed_diff[smoothed_diff > 0].sum()

        smoothed = data_i.norm_membrane.rolling(window=5, center=True, min_periods=5).mean()
        smoothed_diff = smoothed.diff()
        slope_mb = smoothed_diff[smoothed_diff > 0].sum()

        data_sheet.loc[current_mask, "slope_nc"] = slope_nc
        data_sheet.loc[current_mask, "slope_mb"] = slope_mb
        data_sheet.loc[current_mask, "slope_marker"] = slope_mb >= slope_nc
    return data_sheet
