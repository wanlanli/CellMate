import numpy as np


def compute_mask_iou(instance_a: np.ndarray, instance_b: np.ndarray) -> int:
    """
    Computes the Intersection over Union (IoU), Intersection over Area (IoA),
    and Intersection over Area of B (IoB) for two binary masks.

    Parameters:
    -----------
    instance_a : np.ndarray
        Binary mask array for instance A.
    instance_b : np.ndarray
        Binary mask array for instance B.

    Returns:
    --------
    list
        A list containing IoU, IoA, and IoB values.
    """
    intersection = np.count_nonzero(
        np.logical_and(instance_a, instance_b).astype(np.uint8))
    if intersection == 0:
        return [0, 0, 0]
    non_intersection_a = np.count_nonzero(instance_a) - intersection
    non_intersection_b = np.count_nonzero(instance_b) - intersection
    iou = intersection / (intersection + non_intersection_a + non_intersection_b)
    ioa = intersection / (intersection + non_intersection_a)
    iob = intersection / (intersection + non_intersection_b)
    return [iou, ioa, iob]


def is_point_in_polygon(points, polygons):
    points = np.asarray(points)
    result = np.zeros((len(polygons), points.shape[0]), dtype=bool)

    for poly_idx, polygon in enumerate(polygons):
        polygon = np.asarray(polygon)
        num_points = points.shape[0]
        num_vertices = polygon.shape[0]

        inside = np.zeros(num_points, dtype=bool)

        for i in range(num_vertices):
            p1x, p1y = polygon[i]
            p2x, p2y = polygon[(i + 1) % num_vertices]

            min_py = np.minimum(p1y, p2y)
            max_py = np.maximum(p1y, p2y)

            intersect = np.logical_and(points[:, 1] > min_py, points[:, 1] <= max_py)
            cond1 = p1y != p2y
            xinters = (points[:, 1] - p1y) * (p2x - p1x) / (p2y - p1y +0.0001) + p1x
            cond2 = np.logical_or(p1x == p2x, points[:, 0] <= xinters)

            cond = np.logical_and(intersect, np.logical_and(cond1, cond2))
            inside = np.logical_xor(inside, cond)

        result[poly_idx, :] = inside

    return result


def hausdorff_distance(trackers, detections, is_acc=False):
    """
    Calculate the distance matrix between trackers and detections.

    Parameters:
    -----------
    trackers: ndarray
        An array with shape [number of tracked objects x DIMENSION].

    detections: ndarray
        An array with shape [number of new detections x DIMENSION].

    Returns:
    -----------
    distance_matrix: ndarray
        An array with shape (len(trackers), len(detections)) representing the distance matrix
        between the tracked objects and new detections.
    """
    if len(trackers) == 0:
        return None
    distance_matrix = np.square(trackers[:, np.newaxis, :, np.newaxis] - detections[:, np.newaxis]
                                ).sum(axis=-1)

    # aa = np.stack([is_point_in_polygon(detections[i], trackers) for i in range(0, detections.shape[0])], axis=1)
    # bb = np.stack([is_point_in_polygon(trackers[i], detections) for i in range(0, trackers.shape[0])])
    distance_matrix_a = np.min(distance_matrix, axis=2)
    # distance_matrix_a[aa] = 0
    distance_matrix_b = np.min(distance_matrix, axis=3)
    # distance_matrix_b[bb] = 0
    if is_acc:
        tracker_mask = np.stack([is_point_in_polygon(detections[i], trackers) for i in range(0, detections.shape[0])], axis=1)
        instance_mask = np.stack([is_point_in_polygon(trackers[i], detections) for i in range(0, trackers.shape[0])])
        distance_matrix_a[tracker_mask] = 0
        distance_matrix_b[instance_mask] = 0
    distance_matrix_min = np.sqrt(np.minimum(np.max(distance_matrix_a, axis=2),
                                             np.max(distance_matrix_b, axis=2)))
    distance_matrix_max = np.sqrt(np.maximum(np.max(distance_matrix_a, axis=2),
                                             np.max(distance_matrix_b, axis=2)))

    return np.stack([distance_matrix_max, distance_matrix_min])
