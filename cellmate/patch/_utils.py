# Copyright 2024 wlli
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
from scipy.interpolate import interp1d
from scipy.ndimage import  binary_erosion


def move_to_center(point, center=[0, 0], dist=5):
    """
    Moves a point towards a specified center by a given distance.

    Parameters:
    ----------
    point : array-like, shape (n_points, 2)
        The coordinates of the points to be moved.

    center : array-like, shape (2,), optional, default=[0, 0]
        The coordinates of the center point to which the points are being moved.

    dist : float, optional, default=5
        The distance by which the points will move toward the center.

    Returns:
    ----------
    new_points : numpy.ndarray, shape (n_points, 2)
        The new coordinates of the points after moving them towards the center.
    """
    org_length = np.sqrt(np.sum(np.square(point - center), axis=1).astype(np.float16))
    return dist/org_length[:, None] * (center - point)+point


def circle_grid(shape):
    """
    Create a grid of coordinates for a 2D array of the given shape.

    Parameters:
    ----------
    shape : tuple of int
        The shape of the grid, specified as (rows, columns).

    Returns:
    ----------
    X, Y : numpy.ndarray
        Two arrays representing the grid coordinates. `X` contains the row indices, and `Y` contains the column indices.
    """
    X, Y = np.ogrid[:shape[0], :shape[1]]
    return X, Y


def intensity_multiple_points(image, centers, radius, mask, method="mean", percentile=50, background_percentile=50):
    """
    Calculate the mean intensity of circular regions for multiple points in the image, excluding zero values.

    Parameters:
    image (2D array-like): The image from which to calculate the intensity.
    centers (list of tuples): A list of (x, y) coordinates for the center of each circle.
    radius (float): The radius of the circles.

    Returns:
    list: A list of mean intensities for each circular region, excluding zero values.
    """
    # Create a mesh grid with the coordinates of the image
    x, y = circle_grid(image.shape)
    intensities = []
    erosion = binary_erosion(mask, structure=np.ones((radius*2, radius*2)), border_value=True)
    background = np.percentile(image[erosion], background_percentile)

    # np.array([np.mean(self.fluorescence_image[i][:, self.erosion[i] > 0], axis=1)
    #                                                 for i in self.frames])
    for center in centers:
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask_i = (distance <= radius) # & mask
        circular_region = image[mask_i]
        if len(circular_region) > 0:
            if method == "max":
                intensity = circular_region.max()
            elif method == "sum":
                intensity = circular_region.sum()
            elif method == "percentile":
                threshold = np.percentile(circular_region, percentile)
                intensity = circular_region[circular_region > threshold].mean()
            else:
                intensity = circular_region.mean()
        else:
            intensity = 0
        intensities.append(intensity)
    return intensities, background


def center_bg_norm(intensity, bg):
    intensity_norm = intensity - bg  # # /intensity_bg[:, None]
    intensity_norm = intensity_norm.clip(0, None)
    intensity_norm = intensity_norm / np.nanmedian(intensity_norm, where=intensity_norm>0, axis=1, keepdims=1)
    # intensity_norm = intensity_norm / np.nanmedian(intensity_norm, where=intensity_norm>0, axis=1, keepdims=1)
    # intensity_norm = (intensity_norm - intensity_norm.min()) / (intensity_norm.max()-intensity_norm.min())
    return intensity_norm


def centre_points(points: np.ndarray) -> np.ndarray:
    """
    Calculates the centroid of a set of points based on the median position of x and y coordinates.
    This function can be adapted to support other methods of calculating central points.

    Parameters:
    points (np.ndarray): A 2D array where each row represents a point with [x, y] coordinates.

    Returns:
    np.ndarray: A 1D array with the x and y coordinates of the centroid (median position).
    """
    # Calculate the median (50th percentile) for x and y coordinates to determine the centroid
    center_x = np.percentile(points[:, 0], 50)
    center_y = np.percentile(points[:, 1], 50)

    return np.array([center_x, center_y])


def circular_sequence(start: int, end: int, max_id: int, include_end=True):
    """
    Generates a circular sequence from a start to an end index within a circular range.

    Parameters
    ----------
    start : int
        The starting index of the sequence.
    end : int
        The ending index of the sequence.
    max_id : int
        The maximum value in the circular range (inclusive).

    Returns
    -------
    List[int]
        A list representing the circular sequence from start to end.

    Example
    -------
    If max_id = 10, start = 7, and end = 4, the output will be:
    [7, 8, 9, 0, 1, 2, 3]
    """

    sequence = []
    end = end % max_id
    start = start % max_id

    current = start
    while current != end:
        sequence.append(current)
        current = (current + 1) % (max_id)
    if include_end:
        sequence.append(end)  # Include the end point in the sequence
    return sequence


def resample_curve(points: np.ndarray, num_samples: int) -> np.ndarray:
    """
    Resamples a curve represented by a set of points to have evenly spaced points.

    Parameters
    ----------
    points : np.ndarray
        An (n, 2) array of points representing the curve, where each row is [x, y].
    num_samples : int
        The desired number of points in the resampled curve.

    Returns
    -------
    np.ndarray
        An (m, 2) array of resampled points with equal distance between samples.
    """
    # Calculate cumulative distances between consecutive points
    distances = np.sqrt(np.sum(np.diff(points, axis=0)**2, axis=1))
    cumulative_distances = np.insert(np.cumsum(distances), 0, 0)

    # Set up an interpolation function for x and y coordinates
    interp_x = interp1d(cumulative_distances, points[:, 0], kind='linear')
    interp_y = interp1d(cumulative_distances, points[:, 1], kind='linear')

    # Generate equally spaced distances along the curve
    target_distances = np.linspace(0, cumulative_distances[-1], num_samples)

    # Interpolate to get new points
    resampled_points = np.vstack((interp_x(target_distances), interp_y(target_distances))).T

    return resampled_points


def intensity_multiple_points_debug(image, centers, radius, mask, method="mean", percentile=50, background_percentile=50):
    """
    Calculate the mean intensity of circular regions for multiple points in the image, excluding zero values.

    Parameters:
    image (2D array-like): The image from which to calculate the intensity.
    centers (list of tuples): A list of (x, y) coordinates for the center of each circle.
    radius (float): The radius of the circles.

    Returns:
    list: A list of mean intensities for each circular region, excluding zero values.
    """
    # Create a mesh grid with the coordinates of the image
    x, y = circle_grid(image.shape)
    intensities = []
    erosion = binary_erosion(mask, structure=np.ones((radius*2, radius*2)), border_value=True)
    background = np.percentile(image[erosion], background_percentile)

    # np.array([np.mean(self.fluorescence_image[i][:, self.erosion[i] > 0], axis=1)
    #                                                 for i in self.frames])
    mask_i_measure = np.zeros(mask.shape, dtype=np.bool_)
    for center in centers:
        distance = np.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
        mask_i = (distance <= radius) & mask
        circular_region = image[mask_i]
        if len(circular_region) > 0:
            if method == "max":
                intensity = circular_region.max()
            elif method == "sum":
                intensity = circular_region.sum()
            elif method == "percentile":
                threshold = np.percentile(circular_region, percentile)
                intensity = circular_region[circular_region > threshold].mean()
            else:
                intensity = circular_region.mean()
        else:
            intensity = 0
        intensities.append(intensity)
        mask_i_measure[mask_i] = True
    return intensities, background, mask_i_measure


# def get_patches_over_cells(cellnet, image):
#     patch_data = {}
#     for key in cellnet.cells.keys():
#         cell_id = key
#         data_overtime = []
#         bg_overtime = []
#         frames = cellnet.cells[cell_id].frames
#         coords = cellnet.aligned_coords(cell_id)
#         centers = cellnet.center_overtime(cell_id)
#         for i, time in enumerate(frames):
#             coord_t = coords[i]
#             coord_t = move_to_center(coord_t, centers[i], dist=9)
#             data, bg = intensity_multiple_points(image[time, 1], coord_t, 9, (image[time, -1] % 1000 == cell_id), method="mean", background_percentile=50)
#             data_overtime.append(data)
#             bg_overtime.append(bg)
#         data_overtime = np.array(data_overtime)
#         bg_overtime = np.array(bg_overtime)
#         patch = DynamicPatch(data_overtime, bg_overtime)
#         patch_data[cell_id] = patch
#     return patch_data
