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
    background = np.percentile(image[mask], background_percentile)
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
                intensity = np.percentile(circular_region, percentile)
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
