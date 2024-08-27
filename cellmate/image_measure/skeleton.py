import numpy as np
from skimage import measure
from scipy.ndimage import convolve
from scipy.interpolate import splprep, splev
from skimage.measure import find_contours
from skimage.morphology import skeletonize


def skeletonize_cell(cropped_image):
    coords = find_contours(cropped_image)[0]
    skeleton_image = skeletonize(cropped_image)
    skeleton_connected = connected_neighbors(skeleton_image*1)*skeleton_image
    if np.sum(skeleton_connected > 2) > 0:
        skeleton_image[np.where(skeleton_connected>2)] = 0
        labeled_array = measure.label(skeleton_image)
        regions = measure.regionprops(labeled_array)
        areas = [region.area for region in regions]
        max_region = np.argmax(areas)
        skeleton_image = labeled_array == regions[max_region].label

    x, y = np.where(skeleton_image)
    data = connected_neighbors(skeleton_image*1)*skeleton_image
    xtip, ytip = np.where(data == 1)
    points = np.array([x, y]).T
    tips = np.array([xtip, ytip]).T
    index = np.where((points == tips[0]).all(axis=1))[0][0]
    points = np.roll(points, -index, axis=0)
    ordered = nearest_neighbor_ordering(points)
    ordered = smooth_curve(ordered, num_points=20)

    start_point = coords[np.argmin(np.linalg.norm(coords - ordered[0], axis=1))].astype(np.int_)
    end_poinst = coords[np.argmin(np.linalg.norm(coords - ordered[-1], axis=1))].astype(np.int_)
    start_point = find_intersection(ordered[1], ordered[0], coords)
    end_poinst = find_intersection(ordered[-2], ordered[-1], coords)
    if (start_point is not None) & (end_poinst is not None):
        all_points = np.array(list([start_point]) + list(ordered) + list([end_poinst]))
    else:
        all_points = ordered
    return all_points


def connected_neighbors(binary_image):
    # Define the 8-connectivity kernel
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]])
    # Apply convolution to count the neighbors for each pixel
    neighbor_count = convolve(binary_image, kernel, mode='constant', cval=0)
    return neighbor_count


def find_intersection(p1, p2, border_points):
    x1, y1 = p1
    x2, y2 = p2
    intersections = []

    for i in range(len(border_points)):
        x3, y3 = border_points[i]
        x4, y4 = border_points[(i + 1) % len(border_points)]  # Ensure it wraps around to the first point
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if denom == 0:
            continue  # Parallel lines
        # Calculate the intersection point
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denom

        if 0 <= t and 0 <= u <= 1:
            intersect_x = x1 + t * (x2 - x1)
            intersect_y = y1 + t * (y2 - y1)
            intersections.append((intersect_x, intersect_y, t))
    if intersections:
        # Sort by parameter t to find the closest intersection
        intersections.sort(key=lambda x: x[2])
        return intersections[0][:2]  # Return the closest intersection point
    return None


def nearest_neighbor_ordering(points):
    points = points.copy()
    ordered_points = [points[0]]  # Start with the first point
    points = np.delete(points, 0, axis=0)

    while points.shape[0] > 0:
        last_point = ordered_points[-1]
        distances = np.linalg.norm(points - last_point, axis=1)
        nearest_index = np.argmin(distances)
        ordered_points.append(points[nearest_index])
        points = np.delete(points, nearest_index, axis=0)

    return np.array(ordered_points)


def calculate_distance_matrix(points):
    # Convert points to a NumPy array if it's not already
    points = np.array(points)
    # Number of points
    n = points.shape[0]
    # Initialize an n x n distance matrix with zeros
    distance_matrix = np.zeros((n, n))
    # Calculate the distance between each pair of points
    for i in range(n):
        for j in range(i+1, n):
            distance_matrix[i, j] = np.linalg.norm(points[i] - points[j])
            distance_matrix[j, i] = distance_matrix[i, j]
    print(distance_matrix)
    return distance_matrix


def smooth_curve(points, num_points=100):
    # Extract x and y coordinates
    points = np.array(points)
    x = points[:, 0]
    y = points[:, 1]
    # Create a parameter for the curve
    t = np.linspace(0, 1, len(points))
    # Fit a spline to the points
    tck, u = splprep([x, y], s=0)

    # Generate new points along the spline
    u_new = np.linspace(0, 1, num_points)
    x_new, y_new = splev(u_new, tck)

    # Return the new smooth set of points
    smooth_points = np.vstack((x_new, y_new)).T
    return smooth_points
