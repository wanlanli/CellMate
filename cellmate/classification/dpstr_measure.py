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
from typing import Tuple, Optional, Any


import numpy as np
import pandas as pd


from scipy.signal import convolve2d
from skimage.morphology import disk, remove_small_objects, opening, closing, erosion


def dPSTR_image_measure2table(fluorescent_image: np.ndarray,
                              mask: np.ndarray,
                              nucleus_label: np.ndarray,
                              *args: Any,
                              **kwargs: Any) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Measure dPSTR fluorescence intensity statistics for cytoplasm and nucleus
    regions of each labeled cell in an image.

    Workflow
    --------
    - Iterate through each labeled cell in the mask.
    - Detect the nucleus within each cell (refined segmentation).
    - Separate nucleus and cytoplasm regions.
    - Compute mean, max, and min fluorescence values for each region.

    Parameters
    ----------
    fluorescent_image : np.ndarray
        2D array of fluorescence intensity values (dPSTR signal).
    mask : np.ndarray
        2D labeled array (same shape as `fluorescent_image`), where each cell
        has a unique non-zero label. Background is labeled as 0.
    nucleus_label : np.ndarray
        2D array indicating potential nucleus regions.
    *args, **kwargs :
        Additional arguments forwarded to `detect_single_cell_nucleus`
        (e.g., kernel, percentile, selem, area_threshold, erosion_factor).

    Returns
    -------
    table : pd.DataFrame
        DataFrame where each row corresponds to a cell label, with columns:
        - cytoplasm_mean, cytoplasm_max, cytoplasm_min
        - nucleus_mean, nucleus_max, nucleus_min
        - whole_mean, whole_max, whole_min
        - background
    detected_nc : np.ndarray
        Labeled binary image with detected nucleus regions, for visualization
        or debugging.
    """
    # --- Initialize ---
    labels = np.unique(mask)
    labels = labels[labels != 0]

    table = pd.DataFrame(columns=[
        "cytoplasm_mean", "cytoplasm_max", "cytoplasm_min",
        "nucleus_mean", "nucleus_max", "nucleus_min",
        "whole_mean", "whole_max", "whole_min",
        "background"
    ])

    # Background fluorescence statistics
    bg_mean, bg_max, bg_min = __calculate_basic_stats(fluorescent_image[mask == 0])

    detected_nc = np.zeros(mask.shape, dtype=np.uint16)
    # --- Process each cell ---
    for obj_label in labels:
        mask_single = mask == obj_label

        # Skip very small objects
        if mask_single.sum() < 1000:
            continue

        # Detect nucleus within this cell
        nc_detect, detected = detect_single_cell_nucleus(mask_single * nucleus_label, *args, **kwargs)

        # Whole-cell fluorescence stats
        v_mean, v_max, v_min = __calculate_basic_stats(fluorescent_image[mask_single])

        if detected:
            # Nucleus fluorescence stats
            n_mean, n_max, n_min = __calculate_basic_stats(fluorescent_image[nc_detect])

            # Cytoplasm = whole cell - nucleus
            cytoplasm_mask = mask_single & (~nc_detect)
            c_mean, c_max, c_min = __calculate_basic_stats(fluorescent_image[cytoplasm_mask])

            # Store detected nucleus region for this cell
            detected_nc[nc_detect] = (obj_label % 1000) + 9000  # label offset for visualization
        else:
            # No nucleus detected → whole cell is cytoplasm
            n_mean = n_max = n_min = 0
            c_mean, c_max, c_min = v_mean, v_max, v_min

        # Save results
        table.loc[obj_label] = [
            c_mean, c_max, c_min,
            n_mean, n_max, n_min,
            v_mean, v_max, v_min,
            bg_mean
        ]

    return table, detected_nc


def _post_process(image: np.ndarray,
                  selem: Optional[np.ndarray] = None,
                  area_threshold: int = 100,
                  erosion_factor: int = 2):
    """
    Apply morphological post-processing to clean up a binary mask.

    The processing pipeline includes:
    1. Morphological opening to remove small white noise.
    2. Morphological closing to fill small black holes.
    3. Removal of small connected components below a specified area.
    4. Morphological erosion to slightly shrink regions and refine edges.

    Parameters
    ----------
    image : numpy.ndarray
        2D binary input mask (nonzero = foreground).
    selem : numpy.ndarray, optional
        Structuring element for morphological operations.
        If None, a disk of radius 3 is used.
    area_threshold : int, optional
        Minimum pixel area for objects to keep. Smaller objects are removed.
        Default is 100.
    erosion_factor : int, optional
        Radius of erosion (disk size) applied at the final step. Default is 2.

    Returns
    -------
    numpy.ndarray
        Cleaned binary mask (dtype=bool).
    """
    if selem is None:
        selem = disk(3)
    # Morphological opening: remove small white noise
    opened = opening(image, selem)

    # Morphological closing: fill small black holes
    closed = closing(opened, selem)

    # Remove small connected components
    cleaned = remove_small_objects(closed, min_size=area_threshold, connectivity=2)

    # Erode the regions slightly to refine the boundaries
    shrunk = erosion(cleaned, disk(erosion_factor))

    return shrunk


def _detect_nucleus_mask(image: np.ndarray,
                         kernel: Optional[np.ndarray] = None,
                         percentile: int = 90,
                         selem: np.ndarray = disk(3),
                         area_threshold: int = 100,
                         erosion_factor: int = 2):
    """
    Detect nuclear regions from a single-cell image using convolution filtering,
    percentile-based thresholding, and morphological post-processing.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image (e.g., a fluorescence channel highlighting the nucleus).
    kernel : numpy.ndarray, optional
        Convolution kernel to enhance nuclear features.
        If None, a default kernel is used.
    percentile : float, optional
        Percentile (0–100) used to compute the threshold on the convolved response.
        Default is 90.
    selem : numpy.ndarray, optional
        Structuring element for morphological operations.
        Default is disk(3).
    area_threshold : int, optional
        Minimum pixel area for valid objects. Smaller objects are removed.
        Default is 100.
    erosion_factor : int, optional
        Number of erosions applied during post-processing to refine the mask.
        Default is 2.

    Returns
    -------
    numpy.ndarray
        A binary mask (2D array) where detected nucleus regions are True (1)
        and background is False (0).
    """

    # Enhance nuclear signal using convolution with the specified kernel
    if kernel is None:
        kernel = np.array([[0, 0, 1, 0, 0],
                           [0, 1, 2, 1, 0],
                           [1, 2, 4, 2, 1],
                           [0, 1, 2, 1, 0],
                           [0, 0, 1, 0, 0]])
    response = convolve2d(image, kernel, mode='same')

    # Compute threshold value based on the given percentile (ignore background zeros)
    thresh = np.percentile(response[response > 0], percentile)

    # Generate binary image by threshold
    binary = response > thresh

    # Apply morphological post-processing (area filtering, erosion, etc.)
    result = _post_process(binary, selem=selem, area_threshold=area_threshold, erosion_factor=erosion_factor)

    return result


def detect_single_cell_nucleus(image: np.ndarray, *args, **kwargs):
    """
    Detect nuclear regions in a single-cell image with automatic cropping.

    This function crops the input image around the non-zero region, applies nucleus
    detection, and reinserts the result into an output mask of the same size as
    the original image.

    Parameters
    ----------
    image : numpy.ndarray
        2D input image.
    *args, **kwargs :
        Additional arguments forwarded to `_detect_nucleus_mask`, such as:
        - kernel : 2D numpy.ndarray
        - percentile : int
        - selem : numpy.ndarray
        - area_threshold : int
        - erosion_factor : int

    Returns
    -------
    result_mask : numpy.ndarray
        2D binary mask (same shape as input) with detected nucleus regions.
    detected : bool
        True if any nucleus was detected, otherwise False.
    """

    # Find the minimal bounding box of the non-zero region in the image
    x_min, x_max, y_min, y_max = __get_bbox(image)

    # Crop the image to focus on the region of interest
    cropped_image = image[x_min:x_max, y_min:y_max]

    # Apply nucleus detection on the cropped image
    result_cropped = _detect_nucleus_mask(cropped_image, *args, **kwargs)

    # Check if any nucleus was detected
    detected = np.any(result_cropped)

    # Resize result back to original image shape, placing detected region in the original coordinates
    result_resize = np.zeros(image.shape, dtype=np.uint16)
    if detected:
        result_resize[x_min:x_max, y_min:y_max] = result_cropped

    return result_resize.astype(bool), detected


def __calculate_basic_stats(data: np.ndarray) -> Tuple[float, float, float]:
    """
    Calculate basic statistics (mean, maximum, minimum) of the input array.

    Parameters
    ----------
    data : array-like
        Input data, typically a NumPy array or a list of numeric values.

    Returns
    -------
    tuple of float
        (mean, max, min) values of the input data.
    """
    return np.mean(data), np.max(data), np.min(data)


def __get_bbox(data: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute the bounding box of non-zero elements in a 2D array.

    Parameters
    ----------
    data : numpy.ndarray
        A 2D array (e.g., image mask). Non-zero values are considered as the region of interest.

    Returns
    -------
    tuple of int
        (x_min, x_max, y_min, y_max) indices of the bounding box.
    """
    x_index, y_index = np.where(data)
    x_min = x_index.min()
    x_max = x_index.max()
    y_min = y_index.min()
    y_max = y_index.max()
    return x_min, x_max, y_min, y_max
