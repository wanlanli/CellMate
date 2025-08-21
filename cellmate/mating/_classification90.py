import numpy as np
from skimage import morphology
import scipy.ndimage as ndi


def fluorescent_intensity_h90_single(fluorescent_image, mask, erosion_k=13, dilation_k=1, conv_k=17):
    closing = morphology.binary_erosion(mask, footprint=morphology.disk(erosion_k))
    opening = morphology.binary_dilation(mask, footprint=morphology.disk(dilation_k))
    cc = ndi.convolve(closing*fluorescent_image/(conv_k*conv_k), np.ones((conv_k, conv_k)), mode='constant', cval=0)
    idx = np.argmax(cc)
    row, col = np.unravel_index(idx, cc.shape)

    mask_edge = ~closing & opening

    edge_list = fluorescent_image[mask_edge]
    edge_intensity = np.mean(edge_list[edge_list > np.percentile(edge_list, 70)])
    core_intensity = cc[row, col]
    # bg_intensity = np.percentile(fluorescent_image[closing], 10)

    return edge_intensity, core_intensity


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


def intensity_h90_diff_local_single(fluorescent_image, mask, erosion_k=10, threshold=90):
    closed_image = morphology.binary_erosion(mask, footprint=morphology.disk(erosion_k))
    kernel = np.array([[0, 0, 1, 0, 0],
                      [0, 1, 2, 1, 0],
                      [1, 2, 4, 2, 1],
                      [0, 1, 2, 1, 0],
                      [0, 0, 1, 0, 0]])

    response = ndi.convolve((closed_image * fluorescent_image.astype(np.float_)), kernel, mode='constant', cval=0)
    # response = convolve2d(closed_image*f_cropped, kernel, mode='same')

    idx = np.argmax(response)
    row, col = np.unravel_index(idx, response.shape)
    thresh = np.percentile(response[response > 0], threshold)

    # Generate binary image by threshold
    binary = response > thresh

    nc_mask = post_process(binary, area_threshold=200)

    nc_flag = nc_mask[row, col]

    mask_edge = ~closed_image & mask

    edge_list = fluorescent_image[mask_edge]
    edge_intensity = np.mean(edge_list[edge_list > np.percentile(edge_list, 70)])

    if nc_flag:
        core_intensity = np.mean(fluorescent_image[nc_mask])
        cy_intensity = np.mean(fluorescent_image[~nc_mask & closed_image])
    else:
        core_intensity = 0
        cy_intensity = np.mean(fluorescent_image[closed_image])
    # bg_intensity = np.percentile(fluorescent_image[closing], 10)

    return nc_flag, core_intensity, edge_intensity, cy_intensity


def instance_fluorescent_intensity_h90(fluorescent_image, masks, *args, **kwargs):
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
        nc_flag, core_intensity, edge_intensity, cy_intensity = intensity_h90_diff_local_single(f_cropped, mask, *args, **kwargs)
        data = [label, nc_flag, core_intensity, edge_intensity, cy_intensity, bg_intensity]
        data_sheet.append(data)
    return data_sheet
