# Copyright 2025 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright 2025 wlli
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import numpy as np
import pandas as pd


from scipy.signal import convolve2d
from skimage.morphology import disk, remove_small_objects, opening, closing, erosion


def __mean_max_min(data):
    return np.mean(data), np.max(data), np.min(data)


def __cropped_index(data):
    x_index, y_index = np.where(data)
    x_min = x_index.min()
    x_max = x_index.max()
    y_min = y_index.min()
    y_max = y_index.max()
    return x_min, x_max, y_min, y_max


def dPSTR_image_measure2table(florescent_image, mask, nucleus_label, *args, **kwargs):
    """
    Measure dPSTR fluorescence intensity statistics for cytoplasm and nucleus regions of each cell.

    This function:
    - Iterates through each labeled cell in the mask.
    - Detects the nucleus within the cell using a refined segmentation.
    - Separates nucleus and cytoplasm regions.
    - Computes mean, max, and min fluorescence values for each region.
    - Computes background statistics for comparison.

    Parameters:
    -----------
    florescent_image : np.ndarray
        2D array representing the fluorescence intensity (dPSTR signal) of the image.

    mask : np.ndarray
        2D array with the same shape as `florescent_image`, where each cell has a unique non-zero label.
        Background is labeled as 0.

    nucleus_label : np.ndarray
        2D array (binary or labeled) identifying potential nucleus regions.

    *args, **kwargs :
        Passed to `single_cell_nucleus_detect` for customizing detection, such as kernel, percentile, etc.

    Returns:
    --------
    table : pd.DataFrame
        DataFrame where each row corresponds to a cell label, and columns include:
        - cytoplasm_mean, cytoplasm_max, cytoplasm_min
        - nucleus_mean, nucleus_max, nucleus_min
        - whole_mean, whole_max, whole_min (entire cell region)
        - background (mean background fluorescence)

    detected_nc : np.ndarray
        Labeled binary image with detected nucleus regions, used for visualization/debugging.
    """
    # Get all unique labels (excluding background 0)
    labels = np.unique(mask)
    labels = labels[labels != 0]

    # Initialize results table
    table = pd.DataFrame(columns=[
        "cytoplasm_mean", "cytoplasm_max", "cytoplasm_min",
        "nucleus_mean", "nucleus_max", "nucleus_min",
        "whole_mean", "whole_max", "whole_min",
        "background"
    ])

    # Measure background fluorescence in regions where mask == 0
    bg_mean, bg_max, bg_min = __mean_max_min(florescent_image[mask == 0])

    # For storing detected nucleus labels (for debugging or visualization)
    detected_nc = np.zeros(mask.shape, dtype=np.uint16)

    for obj_label in labels:
        # Mask for a single cell
        mask_single = mask == obj_label

        # Try to detect nucleus in this cell
        nc_detect, flag = single_cell_nucleus_detect(mask_single * nucleus_label, *args, **kwargs)
        nc_mask = nc_detect > 0

        # Whole-cell fluorescence stats
        v_mean, v_max, v_min = __mean_max_min(florescent_image[mask_single])

        if flag:
            # Nucleus fluorescence stats
            n_mean, n_max, n_min = __mean_max_min(florescent_image[nc_mask])

            # Cytoplasm region is the cell minus nucleus
            cytoplasm_mask = mask_single & (~nc_mask)
            c_mean, c_max, c_min = __mean_max_min(florescent_image[cytoplasm_mask])

            # Store detected nucleus region for this cell
            detected_nc[nc_mask] = (obj_label % 1000) + 9000  # label offset for visualization
        else:
            # If no nucleus detected, treat entire cell as cytoplasm
            n_mean = n_max = n_min = 0
            c_mean, c_max, c_min = v_mean, v_max, v_min

        # Add measurement row to results table
        table.loc[obj_label] = [
            c_mean, c_max, c_min,
            n_mean, n_max, n_min,
            v_mean, v_max, v_min,
            bg_mean
        ]

    return table, detected_nc


def post_process(image, selem=disk(3), area_threshold=100, erosion_factor=2):
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
    opened = opening(image, selem)

    # Morphological closing: fill small black holes
    closed = closing(opened, selem)

    # Remove small connected components
    img_rm_small = remove_small_objects(closed, min_size=area_threshold, connectivity=2)

    # Erode the regions slightly to refine the boundaries
    shrunk = erosion(img_rm_small, disk(erosion_factor))

    return shrunk


def _single_cell_nucleus_detect(image,
                                kernel=np.array([[0, 0, 1, 0, 0],
                                                 [0, 1, 2, 1, 0],
                                                 [1, 2, 4, 2, 1],
                                                 [0, 1, 2, 1, 0],
                                                 [0, 0, 1, 0, 0]]),
                                percentile=90,
                                selem=disk(3),
                                area_threshold=100,
                                erosion_factor=2):
    """
    Detect nucleus regions from a single-cell image using convolution and post-processing.

    Parameters:
    ----------
    image : 2D numpy array
        Input image (e.g. fluorescence channel with nuclear marker).
    kernel : 2D numpy array, optional
        Convolution kernel to enhance nuclear features. Default is a Gaussian-like kernel.
    percentile : float, optional
        Percentile threshold to binarize the convolved response. Default is 90.
    selem : ndarray, optional
        Structuring element used in morphological operations. Default is disk(3).
    area_threshold : int, optional
        Minimum area for valid segmented objects. Small objects are removed. Default is 100.
    erosion_factor : int, optional
        Number of erosion steps during post-processing to refine shapes. Default is 2.

    Returns:
    -------
    result : 2D binary numpy array
        Binary mask indicating detected nucleus regions.
    """

    # Enhance nuclear signal using convolution with the specified kernel
    response = convolve2d(image, kernel, mode='same')

    # Compute threshold value based on the given percentile (ignore background zeros)
    thresh = np.percentile(response[response > 0], percentile)

    # Generate binary image by threshold
    binary = response > thresh

    # Apply morphological post-processing (area filtering, erosion, etc.)
    result = post_process(binary, selem=selem, area_threshold=area_threshold, erosion_factor=erosion_factor)

    return result


def single_cell_nucleus_detect(image, *args, **kwargs):
    """
    Wrapper for nucleus detection in a single-cell image with automatic cropping.

    This function crops the image around the region of interest (non-zero area),
    applies nucleus detection, and places the result back into the original image size.

    Parameters:
    ----------
    image : 2D numpy array
        The input image (e.g., nuclear fluorescence channel) for a single cell.
    *args, **kwargs :
        Additional arguments passed to `_single_cell_nucleus_detect`, such as
        kernel, percentile, selem, area_threshold, and erosion_factor.

    Returns:
    -------
    result_resize : 2D numpy array (same shape as input)
        Binary mask of detected nucleus, placed into the original image dimensions.
    flag : bool
        True if any nucleus was detected, otherwise False.
    """

    # Find the minimal bounding box of the non-zero region in the image
    x_min, x_max, y_min, y_max = __cropped_index(image)

    # Crop the image to focus on the region of interest
    cropped_image = image[x_min:x_max, y_min:y_max]

    # Apply nucleus detection on the cropped image
    result = _single_cell_nucleus_detect(cropped_image, *args, **kwargs)

    # Check if any nucleus was detected
    flag = np.any(result)

    # Resize result back to original image shape, placing detected region in the original coordinates
    result_resize = np.zeros(image.shape, dtype=np.uint16)
    if flag:
        result_resize[x_min:x_max, y_min:y_max] = result

    return result_resize, flag
