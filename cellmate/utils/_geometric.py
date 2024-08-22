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


def angle_of_vectors(v1, v2):
    """
    Calculates the angle between two 2D vectors, v1 and v2, in radians.
    The angle is measured in the counterclockwise direction from v1 to v2
    and is normalized to the range [0, 2π).

    Parameters:
    v1 (array-like): The first vector, represented as [x1, y1].
    v2 (array-like): The second vector, represented as [x2, y2].

    Returns:
    float: The angle between v1 and v2 in radians, normalized to [0, 2π).
    """
    # Calculate the angle using arctan2, which handles sign and quadrant automatically
    angle_v1 = np.arctan2(v1[1], v1[0])
    angle_v2 = np.arctan2(v2[1], v2[0])

    # Compute the difference between the two angles
    angle_difference = angle_v2 - angle_v1
    # Normalize the angle to be within [0, 2π)
    angle_difference = angle_difference % (2 * np.pi)

    return angle_difference


def create_line(point1, point2, steps: int = 5):
    """
    Draws a line from point1 to point2 and resamples points along the line, excluding the endpoints.

    Parameters:
    point1 (array-like): The starting point of the line (e.g., [x1, y1]).
    point2 (array-like): The ending point of the line (e.g., [x2, y2]).
    steps (int, optional): The desired spacing between resampled points along the line. Default is 5.

    Returns:
    np.ndarray: A 2D array of resampled points along the line, excluding the endpoints.
                The array has a shape of (2, number_of_points), where the first row contains x-coordinates
                and the second row contains y-coordinates.
    """
    # Calculate the Euclidean distance between point1 and point2.
    distance = np.sqrt(np.sum(np.square(point1 - point2)))
    # Determine the number of points to generate along the line, based on the specified step size.
    number_of_points = int(np.ceil(distance / steps))

    x = np.linspace(point1[0], point2[0], number_of_points)[1:-1]
    y = np.linspace(point1[1], point2[1], number_of_points)[1:-1]
    return np.array([x, y], dtype=np.int_)


def included_angle(angle):
    """
    Reduces the given angle to its corresponding acute angle, mapping it to the range [0, π/2].

    Parameters:
    angle (float): The angle in radians to be reduced.

    Returns:
    float: The reduced angle, mapped to the range [0, π/2].
    """
    # Normalize the angle to the range [0, 2π)
    angle = angle % (2 * np.pi)

    # Map the angle to the corresponding acute angle in the range [0, π/2]
    if angle < np.pi/2:
        return angle
    elif angle < np.pi:
        return np.pi - angle
    elif angle < 3*np.pi/2:
        return angle - np.pi
    else:
        return 2*np.pi - angle
