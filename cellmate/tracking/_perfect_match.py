import numpy as np


def match_cells_generation_iou(distance, threshold, unmatched_trackers, unmatched_instance):
    """
    Perform matching of cells across generations using the provided distance matrix and threshold.

    Parameters:
    -----------
    distance : np.array
        A 3D NumPy array with shape (len(tracker), len(instance), 3) where each element represents
        the distance metrics [iou, ioa, iob] between a tracker and an instance.
    threshold : float
        The threshold value to determine if a distance is considered a match.
    unmatched_trackers : list
        A list of indices of trackers that did not match any instance initially.
    unmatched_instance : list
        A list of indices of instances that did not match any tracker initially.

    Returns:
    --------
    tuple
        Contains four elements:
        1. matched_f (list): List of [tracker1, tracker2, instance] indices for fusion matches.
        2. matched_d (list): List of [tracker, instance1, instance2] indices for division matches.
        3. unmatched_trackers (list): List of indices of trackers that did not match with any instance.
        4. unmatched_instance (list): List of indices of instances that did not match with any tracker.
    """
    # matched_indices, unmatched_trackers, unmatched_instance = one_to_one_match(distance, threshold)

    # Perform fusion matching
    if ((len(unmatched_trackers) == 0) | (len(unmatched_instance) == 0)):
        return ([], [], unmatched_trackers, unmatched_instance)
    fusion_data = distance[:, :, 1].copy()
    unmatched_distance = fusion_data[unmatched_trackers]
    unmatched_distance = unmatched_distance[:, unmatched_instance]
    matched_f, unmatched_trackers_f, unmatched_instance_f = two_to_one_match(unmatched_distance > threshold)

    # Map back to original indices
    matched_f = [[unmatched_trackers[i[0]],
                  unmatched_trackers[i[1]],
                  unmatched_instance[i[2]]] for i in matched_f]

    # Update unmatched trackers and instances after fusion
    unmatched_trackers = [unmatched_trackers[i] for i in unmatched_trackers_f]
    unmatched_instance = [unmatched_instance[i] for i in unmatched_instance_f]

    if ((len(unmatched_trackers) == 0) | (len(unmatched_instance) == 0)):
        return (matched_f, [], unmatched_trackers, unmatched_instance)
    # Perform division matching
    division_data = distance[:, :, 2].copy()
    unmatched_distance = division_data[unmatched_trackers]
    unmatched_distance = unmatched_distance[:, unmatched_instance]
    matched_d, unmatched_trackers_d, unmatched_instance_d = one_to_two_match(unmatched_distance > threshold)

    # Map back to original indices
    matched_d = [[unmatched_trackers[i[0]],
                 unmatched_instance[i[1]],
                 unmatched_instance[i[2]]] for i in matched_d]

    # Update unmatched trackers and instances after division
    unmatched_trackers = [unmatched_trackers[i] for i in unmatched_trackers_d]
    unmatched_instance = [unmatched_instance[i] for i in unmatched_instance_d]
    return (matched_f, matched_d, unmatched_trackers, unmatched_instance)


def match_cells_generation_hausdorff(distance_matrix, unmatched_trackers, unmatched_instance, tra_threshold, 
                                     det_threshold, high_ration):
    """
    Calculate generation relationships (e.g., fusion, division) based on a distance matrix.
    Three grouped objects are accepted if the distance is within a given ratio.

    Parameters:
    -----------
    distance_matrix: ndarray
        An array with shape (len(trackers), len(detections)) representing the distance matrix
        between the tracked objects and new detections.

    unmatched_trackers: list
        A list of unmatched trackers that were not accepted.

    unmatched_detections: list
        A list of unmatched new detections that were not accepted.

    tra_threshold: ndarray
        An array with shape (len(unmatched_trackers), 2) containing thresholds obtained from the unmatched tracked objects.

    det_threshold: ndarray
        An array with shape (len(unmatched_detections), 2) containing thresholds obtained from the unmatched new detected objects.

    high_ratio: float
        The accepted distance error ratio between tracked objects and new detected objects.

    Returns:
    -----------
    matched_fusion: list of tuples
        A list of tuples representing matched fusion pairs of indices (tracker_index1, tracker_index2, detection_index).
        This indicates that tracker_index1 and tracker_index2 have fused into detection_index.

    matched_division: list of tuples
        A list of tuples representing matched division pairs of indices (tracker_index1, detection_index1, detection_index2).
        This indicates that tracker_index1 has divided into detection_index1 and detection_index2.

    unmatched_trackers: list
        A list of unmatched trackers that were not accepted.

    unmatched_detections: list
        A list of unmatched new detections that were not accepted.
    """
    untraced_dist = distance_matrix[unmatched_trackers]
    untraced_dist = untraced_dist[:, unmatched_instance]
    unmatched_binary = (untraced_dist < tra_threshold[unmatched_trackers, 0, None]*high_ration) \
                         & (untraced_dist < det_threshold[None, unmatched_instance, 0]*high_ration)
    # unmatched_binary = untraced_dist < 20
    # detect division
    matched_d, unmatched_trackers_d, unmatched_instance_d = one_to_two_match(unmatched_binary)
    matched_d = [[unmatched_trackers[i[0]],
                    unmatched_instance[i[1]],
                    unmatched_instance[i[2]]] for i in matched_d]

    # Update unmatched trackers and instances after division
    unmatched_trackers = [unmatched_trackers[i] for i in unmatched_trackers_d]
    unmatched_instance = [unmatched_instance[i] for i in unmatched_instance_d]

    untraced_dist = distance_matrix[unmatched_trackers]
    untraced_dist = untraced_dist[:, unmatched_instance]
    unmatched_binary = (untraced_dist < tra_threshold[unmatched_trackers, 0, None]*high_ration) \
                        & (untraced_dist < det_threshold[None, unmatched_instance, 0]*high_ration)

    # detect fusion
    matched_f, unmatched_trackers_f, unmatched_instance_f = two_to_one_match(unmatched_binary)

    matched_f = [[unmatched_trackers[i[0]],
                  unmatched_trackers[i[1]],
                  unmatched_instance[i[2]]] for i in matched_f]

    # Update unmatched trackers and instances after fusion
    unmatched_trackers = [unmatched_trackers[i] for i in unmatched_trackers_f]
    unmatched_instance = [unmatched_instance[i] for i in unmatched_instance_f]

    return matched_f, matched_d, unmatched_trackers, unmatched_instance


def one_to_one_match(distance_binary: np.array):
    """
    Determine one-to-one matches based on a binary distance matrix.

    Parameters:
    -----------
    distance_binary : np.array
        A 2D NumPy array where each element is a binary value indicating whether.
        the distance between a tracker (row) and an instance (column) meets a threshold.

    Returns:
    --------
    tuple
        Contains three elements:
        1. matched_indices (np.array): An array of [row, column] indices where each pair meets the one-to-one match criteria.
        2. unmatched_trackers (list): List of indices of trackers that did not match any instance.
        3. unmatched_instances (list): List of indices of instances that did not match any tracker.

    This function ensures that each tracker (row) and each instance (column)
    are matched uniquely to one counterpart based on the binary distance matrix.
    """
    matched_binary = distance_binary.copy()
    matched_binary[distance_binary.sum(1) != 1] = 0
    matched_binary[:, distance_binary.sum(0) != 1] = 0
    if matched_binary.sum() > 0:
        matched_indices = np.stack(np.where(matched_binary), axis=1)
    else:
        matched_indices = np.empty(shape=(0, 2))

    unmatched_trackers = [i for i in range(0, distance_binary.shape[0]) if i not in matched_indices[:, 0]]
    unmatched_instance = [i for i in range(0, distance_binary.shape[1]) if i not in matched_indices[:, 1]]

    return (matched_indices, unmatched_trackers, unmatched_instance)


def two_to_one_match(distance_binary):
    """
    Identify two-to-one matches based on a binary distance matrix, where each instance can
    match with exactly two trackers.

    Parameters:
    -----------
    distance_binary : np.array
        A 2D NumPy array where each element is a binary value indicating whether 
        the distance between a tracker (row) and an instance (column) meets a threshold.

    Returns:
    --------
    tuple
        Contains three elements:
        1. matched (np.array): An array of [tracker1, tracker2, instance] indices for each match that meets
           the two-to-one match criteria.
        2. unmatched_trackers (list): List of indices of trackers that did not match with any instance.
        3. unmatched_instances (list): List of indices of instances that did not match with any two trackers.

    This function processes a distance matrix to find instances where exactly two trackers have distances
    below the given threshold to a single instance. These matches must also ensure that each tracker is
    uniquely matched to only this instance for a valid match.
    """
    matched = []

    # Identify columns where the sum is exactly 2, indicating a potential match
    col = np.where(distance_binary.sum(0) == 2)[0]
    for i in col:
        row = np.where(distance_binary[:, i])[0]
        if (distance_binary[row].sum(1) == [1, 1]).all():
            matched.append([row[0], row[1], i])
    if len(matched) > 0:
        matched = np.array(matched)
        # print("fusion:", matched)
    else:
        matched = np.empty(shape=(0, 3))
    unmatched_trackers = [i for i in range(0, distance_binary.shape[0]) if i not in matched[:, 0:2]]
    unmatched_instance = [i for i in range(0, distance_binary.shape[1]) if i not in matched[:, 2]]

    return (matched, unmatched_trackers, unmatched_instance)


def one_to_two_match(distance_binary):
    """
    Identify one-to-two matches based on a binary distance matrix, where each tracker can
    match with exactly two instances.

    Parameters:
    -----------
    distance_binary : np.array
        A 2D NumPy array where each element is a binary value indicating whether 
        the distance between a tracker (row) and an instance (column) meets a threshold.

    Returns:
    --------
    tuple
        Contains three elements:
        1. matched (np.array): An array of [tracker, instance1, instance2] indices for each match that meets
           the one-to-two match criteria.
        2. unmatched_trackers (list): List of indices of trackers that did not match with any two instances.
        3. unmatched_instances (list): List of indices of instances that did not match with any tracker.

    This function processes a distance matrix to find trackers where exactly two instances have distances
    below the given threshold to a single tracker. These matches must also ensure that each instance is
    uniquely matched to only this tracker for a valid match.
    """
    matched = []
    row = np.where(distance_binary.sum(1) == 2)[0]
    for i in row:
        col = np.where(distance_binary[i])[0]
        if (distance_binary[:, col].sum(0) == [1, 1]).all():
            matched.append([i, col[0], col[1]])

    if len(matched) > 0:
        matched = np.array(matched)
        # print("divison:", matched)
    else:
        matched = np.empty(shape=(0, 3))
    unmatched_trackers = [i for i in range(0, distance_binary.shape[0]) if i not in matched[:, 0]]
    unmatched_instance = [i for i in range(0, distance_binary.shape[1]) if i not in matched[:, 1:]]
    return (matched, unmatched_trackers, unmatched_instance)


def wash_distance_with_golden(distance, golden_threshold=0.8):
    """
    Modifies the distance matrix by applying the golden threshold. Rows and columns corresponding
    to distances that meet or exceed the threshold are zeroed out, except at the intersections where
    the distance is set to 1.

    Parameters:
    -----------
    distance : numpy.ndarray
        A 2D array (matrix) of distances.

    golden_threshold : float, optional
        The threshold value. Distances greater than or equal to this threshold will trigger the row/column washing.
        Default is 0.8.

    Returns:
    --------
    numpy.ndarray
        A modified version of the input distance matrix, with rows and columns zeroed out based on the threshold,
        and specific positions set to 1 where both row and column meet the threshold.
    """
    rows, cols = np.where(distance >= golden_threshold)
    golden_dist = distance.copy()
    golden_dist[rows, :] = 0
    golden_dist[:, cols] = 0
    golden_dist[rows, cols] = 1
    return golden_dist
