import numpy as np

from scipy import ndimage
from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array

from ._find_contours import find_contours

from ._skeletonize import skeletonize, medial_axis


def skeletonize_cell(image):
    """
    Skeletonizes a binary image of a cell and then refines the skeleton based on a given coordinate.

    Parameters:
    -----------
    image : numpy.ndarray
        A 2D binary image where the cell is represented by white pixels (value 1) and the background 
        by black pixels (value 0).

    Returns:
    --------
    numpy.ndarray
        A 2D array representing the refined skeletonized line based on the given coordinates.
    """
    sk_image = skeletonize_image(image)
    if sk_image.sum() < 2:
        return None
    sk_no_branch = remove_skeleton_branch(sk_image)
    path = find_skeleton_route(sk_no_branch)
    return path


def skeletonize_image(cropped_image, method: str = "lee"):
    """
    Skeletonizes a binary image using the specified method.

    Parameters:
    -----------
    cropped_image : numpy.ndarray
        A 2D binary image where the object of interest is represented by white pixels (value 1)
        and the background by black pixels (value 0).

    method : str, optional
        The method used for skeletonization. Options are:
        - "lee": Uses the Lee method for skeletonization (default).
        - "medial_axis": Uses the medial axis method.
        - Any other value will default to using the standard skeletonize method.

    Returns:
    --------
    numpy.ndarray
        A 2D array representing the skeletonized version of the input image.
    """
    if method == "lee":
        skeleton_image = skeletonize(cropped_image, method="lee")
    elif method == "medial_axis":
        skeleton_image = medial_axis(cropped_image)
    else:
        skeleton_image = skeletonize(cropped_image)
    return skeleton_image


def remove_skeleton_branch(image):
    """
    Removes branches from a skeletonized image, leaving only the main structure.

    Parameters:
    -----------
    image : numpy.ndarray
        A 2D binary image where the skeleton is represented by white pixels (value 1) and the background
        by black pixels (value 0).

    Returns:
    --------
    numpy.ndarray
        A 2D binary image with branches removed, leaving only the main skeleton structure.
    """
    connected = connected_neighbors(image*1)*image
    connected[np.where(connected > 2)] = 0
    labeled, num = ndimage.label(connected, structure=np.ones((3, 3)))
    lbls = np.arange(1, num+1)
    areas = ndimage.labeled_comprehension(connected, labeled, lbls, np.sum, float, 0)
    max_region_label = lbls[np.argmax(areas)]
    max_region = labeled == max_region_label
    return connected_neighbors(max_region)


def connected_neighbors(binary_image):
    """
    Calculates the number of connected neighbors for each pixel in a binary image using 8-connectivity.

    Parameters:
    -----------
    binary_image : numpy.ndarray
        A 2D binary image where each pixel has a value of either 0 or 1.
        The image is represented as a NumPy array.

    Returns:
    --------
    numpy.ndarray
        A 2D array of the same shape as `binary_image`, where each element
        represents the number of 8-connected neighbors of the corresponding pixel in the input image.
    """
    # Define the 8-connectivity kernel (3x3 matrix) where the center pixel is excluded.
    # The kernel will consider the 8 surrounding pixels for each pixel in the binary image.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    # Apply convolution to count the neighbors for each pixel
    neighbor_count = ndimage.convolve(binary_image*1, kernel, mode='constant', cval=0)
    return neighbor_count*binary_image


def find_skeleton_route(image):
    """
    Finds the shortest route between two endpoints on a skeletonized image.

    Parameters:
    -----------
    image : numpy.ndarray
        A 2D binary image where the skeleton is represented by white pixels (value 1) and the background 
        by black pixels (value 0).

    Returns:
    --------
    numpy.ndarray or None
        A 2D array representing the coordinates of the path between the two endpoints on the skeleton.
        Returns None if there are not exactly two endpoints.
    """
    x_tip, y_tip = np.where(image == 1)
    if len(x_tip) != 2:
        print("NUmber of branch point != 2")
        return None
    points = np.array([x_tip, y_tip]).T
    path, _ = route_through_array(image < 1, points[0], points[1], geometric=False)
    return np.array(path)


def find_tips_axis(path, boarder):
    """
    Extends a path by adding intersection points with the border at both ends, if they exist.

    Parameters:
    -----------
    path : numpy.ndarray
        A 2D array of shape (n, 2), where each row represents a point (x, y) along the path.

    border : list of tuples
        A list of (x, y) coordinates defining the border polygon.

    Returns:
    --------
    numpy.ndarray
        The extended path including the intersection points, or the original path if no intersections are found.
    """
    start_point = find_intersection(path[1], path[0], boarder)
    if start_point is not None:
        path[0] = start_point
    end_point = find_intersection(path[-2], path[-1], boarder)
    if end_point is not None:
        path[-1] = end_point  # np.array(list([end_point]) + list(path))
        return path
    else:
        print("No tips found!")
        return path


def find_tips(path, boarder):
    """
    Extends a path by adding intersection points with the border at both ends, if they exist.

    Parameters:
    -----------
    path : numpy.ndarray
        A 2D array of shape (n, 2), where each row represents a point (x, y) along the path.

    border : list of tuples
        A list of (x, y) coordinates defining the border polygon.

    Returns:
    --------
    numpy.ndarray
        The extended path including the intersection points, or the original path if no intersections are found.
    """
    start_point = find_intersection(path[1], path[0], boarder)
    if start_point is not None:
        path = list([start_point]) + list(path)

    end_point = find_intersection(path[-2], path[-1], boarder)
    if end_point is not None:
        path = list(path) + list([end_point])

    if (start_point is None) & (end_point is None):
        print("No tips found!")
    return np.array(path)


def smooth_curve(points, num_points=100, s=100):
    """
    Generates a smooth curve through a set of input points using spline interpolation.

    Parameters:
    -----------
    points : array-like
        A 2D array or list of shape (n, 2), where each row represents a point with (x, y) coordinates.

    num_points : int, optional
        The number of points to generate along the smooth curve. Default is 100.

    s : float, optional
        Smoothing factor used by the spline fitting function. Default is 100. Lower values make the spline fit closer to the original points.

    Returns:
    --------
    numpy.ndarray
        A 2D array of shape (num_points, 2) containing the (x, y) coordinates of the smoothed curve.
    """
    # Create a parameter for the curve
    # t = np.linspace(0, 1, len(points))
    # Fit a spline to the points
    if points.shape[0] <= 15:
        tck, u = splprep([points[:, 0], points[:, 1]], k=1, s=10)
    else:
        tck, u = splprep([points[:, 0], points[:, 1]], k=2, s=s)
    # Generate new points along the spline
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    # Return the new smooth set of points
    smooth_points = np.vstack((x_new, y_new)).T
    return smooth_points


def find_intersection(p1, p2, border_points):
    """
    Finds the intersection point between a line segment (p1, p2) and a polygon
    defined by border_points.

    Parameters:
    -----------
    p1 : tuple of float
        The (x, y) coordinates of the first point of the line segment.

    p2 : tuple of float
        The (x, y) coordinates of the second point of the line segment.

    border_points : list of tuples of float
        A list of (x, y) coordinates that define the vertices of a polygon.
        The polygon is assumed to be closed, meaning that the last vertex is
        implicitly connected to the first vertex.

    Returns:
    --------
    tuple of float or None
        The (x, y) coordinates of the closest intersection point between the 
        line segment and the polygon. If no intersection is found, returns None.
    """
    x1, y1 = p1
    x2, y2 = p2
    direction_vector = np.array(p2) - np.array(p1)
    intersections = []

    for i in range(len(border_points)):
        x3, y3 = border_points[i]
        x4, y4 = border_points[(i + 1) % len(border_points)]  # Ensure it wraps around to the first point
        intersection = _line_intersection(x1, y1, x2, y2, x3, y3, x4, y4)
        if intersection is not None:
            if np.dot(np.array(intersection)[:2] - np.array(p1), direction_vector) > 0:
                intersections.append(intersection)
            # intersections.append(intersection)
    if intersections:
        # Sort by parameter t to find the closest intersection
        intersections.sort(key=lambda x: x[2])
        return intersections[0][:2]  # Return the closest intersection point
    return None


def _line_intersection(x1, y1, x2, y2, x3, y3, x4, y4):
    """
    Finds the intersection point (if any) between the line segment p1p2 and p3p4.

    Parameters:
    -----------
    x1, y1 : float
        The coordinates of the first point of the first line segment (p1).

    x2, y2 : float
        The coordinates of the second point of the first line segment (p2).

    x3, y3 : float
        The coordinates of the first point of the second line segment (p3).

    x4, y4 : float
        The coordinates of the second point of the second line segment (p4).

    Returns:
    --------
    tuple or None
        A tuple (intersect_x, intersect_y, t) representing the coordinates of the intersection point
        and the parameter t that represents the position of the intersection along the first line segment.
        If there is no intersection, returns None.
    """
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if denom == 0:
        return None
    # Calculate the intersection point
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

    if 0 <= t and 0 <= u <= 1:
        intersect_x = x1 + t * (x2 - x1)
        intersect_y = y1 + t * (y2 - y1)
        return (intersect_x, intersect_y, t)
    else:
        return None
        # intersections.append((intersect_x, intersect_y, t))


def find_contours_smooth(image, smooth_point: int = 60, *args, **kwarg):
    """
    Finds contours in a binary image and smooths the first contour found using spline interpolation.

    Parameters:
    -----------
    image : array-like
        A 2D binary image (or any suitable image) where contours are to be found.

    smooth_point : int, optional
        The number of points to generate along the smoothed contour. Default is 100.

    *args : tuple
        Additional positional arguments to be passed to the `find_contours` function.

    **kwarg : dict
        Additional keyword arguments to be passed to the `find_contours` function.

    Returns:
    --------
    numpy.ndarray or None
        A 2D array of shape (smooth_point, 2) containing the smoothed contour points, 
        or None if no contours are found.
    """
    contours = find_contours(image=image, *args, **kwarg)
    if len(contours) > 0:
        contours_smoothed = smooth_curve(contours[0], smooth_point, s=20)
        return contours_smoothed
    else:
        return None


def perpendicular(p1, p2, coords):
    """
    Finds the points where the perpendicular bisector of the line segment p1p2 intersects
    with the boundary defined by `coords`.

    Parameters:
    -----------
    p1, p2 : tuple of float
        The (x, y) coordinates of two points defining the original line segment.

    coords : list of tuples
        A list of (x, y) coordinates representing a polygon or boundary for intersection testing.

    Returns:
    --------
    numpy.ndarray or None
        A 2D array with the coordinates of the two intersection points, or None if no intersections are found.
    """
    a, m, b = perpendicular_line_through_midpoint(p1, p2)
    start_a = find_intersection(m, a, coords)
    start_b = find_intersection(m, b, coords)
    if (start_a is None) or (start_b is None):
        return None
    return np.array([start_a, start_b])


def perpendicular_grid(skeleton, coords):
    """
    Generates a grid of perpendicular lines to segments of a skeleton that intersect with a boundary.

    Parameters:
    -----------
    skeleton : numpy.ndarray
        A 2D array of shape (n, 2) where each row represents a point (x, y) on the skeleton.

    coords : list of tuples
        A list of (x, y) coordinates representing a polygon or boundary for intersection testing.

    Returns:
    --------
    numpy.ndarray
        A 3D array of shape (m, 2, 2), where m is the number of perpendicular lines found. Each 2D array 
        represents two points (x, y) defining a perpendicular line that intersects with the boundary.
    """
    grid = []
    for i in range(0, len(skeleton)-1):
        data = perpendicular(skeleton[i], skeleton[i+1], coords)
        if data is not None:
            grid.append(data)
    return np.array(grid)


def perpendicular_line_through_midpoint(p1, p2):
    """
    Calculates two points that define a line perpendicular to the line segment p1p2
    and passing through the midpoint of p1p2.

    Parameters:
    -----------
    p1, p2 : tuple of float
        The (x, y) coordinates of the two points defining the original line segment.

    Returns:
    --------
    tuple of tuples
        Two points (x3, y3) and (x4, y4) that define the perpendicular line through the midpoint of p1p2.
    """
    x1 = p1[0]
    y1 = p1[1]

    x2 = p2[0]
    y2 = p2[1]
    # Calculate slope of the original line
    if x2 - x1 != 0:
        m = (y2 - y1) / (x2 - x1)
    else:
        # Handle the case where the line is vertical (slope is undefined)
        m = np.inf

    # Calculate the perpendicular slope
    if m != 0 and m != np.inf:
        m_perpendicular = -1 / m
    else:
        # Handle the case where the original line is horizontal (m = 0) or vertical (m = inf)
        m_perpendicular = 0 if m == np.inf else np.inf

    # Calculate the midpoint
    xm = (x1 + x2) / 2
    ym = (y1 + y2) / 2

    # Define two points on the perpendicular line
    delta_x = 1
    if m_perpendicular != np.inf:
        # Choose two x-values near the midpoint and calculate corresponding y-values
        x3 = xm - delta_x
        y3 = m_perpendicular * (x3 - xm) + ym

        x4 = xm + delta_x
        y4 = m_perpendicular * (x4 - xm) + ym
    else:
        # If the perpendicular line is vertical, we choose the same x-values and different y-values
        x3, x4 = xm, xm
        y3, y4 = ym - delta_x, ym + delta_x
    return (x3, y3), (xm, ym), (x4, y4)
