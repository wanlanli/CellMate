# import numpy as np
# import math
# from scipy.interpolate import interp1d


# def find_start_angle(target_angle, point_list):
#     """Find the start point based on degree of major axis.
#     """
#     angles = np.array([angle_of_vectors(point_list[:, i], np.array([1, 0]))
#                        for i in range(0, point_list.shape[1])])
#     # print(point_list.shape)
#     # angles = angle_between_points_array(
#     #     point_list, np.array([[1, 0]])
#     # )
#     target_angle_index = np.argmin(abs(angles-target_angle) % (np.pi*2))
#     return target_angle_index


# def included_angle(angle):
#     # angle = (source - target) % (np.pi*2)
#     if angle < np.pi/2:
#         angle = angle
#     elif (angle > np.pi/2) & (angle < np.pi):
#         angle = np.pi - angle
#     elif (angle > np.pi) & (angle < np.pi*3/2):
#         angle = angle - np.pi
#     elif (angle > np.pi*3/2) & (angle < np.pi*2):
#         angle = np.pi*2 - angle
#     return angle


# def rotation_to_degree(roation):
#     return roation*180/np.pi


# def _validate_vector(u, dtype=None):
#     # XXX Is order='c' really necessary?
#     u = np.asarray(u, dtype=dtype, order='c')
#     if u.ndim == 1:
#         return u
#     raise ValueError("Input vector should be 1-D.")


# def _validate_weights(w, dtype=np.double):
#     w = _validate_vector(w, dtype=dtype)
#     if np.any(w < 0):
#         raise ValueError("Input weights should be all non-negative")
#     return w


# def correlation(u, v, w=None, centered=True):
#     """
#     Compute the correlation distance between two 1-D arrays.

#     The correlation distance between `u` and `v`, is
#     defined as

#     .. math::

#         1 - \\frac{(u - \\bar{u}) \\cdot (v - \\bar{v})}
#                   {{\\|(u - \\bar{u})\\|}_2 {\\|(v - \\bar{v})\\|}_2}

#     where :math:`\\bar{u}` is the mean of the elements of `u`
#     and :math:`x \\cdot y` is the dot product of :math:`x` and :math:`y`.

#     Parameters
#     ----------
#     u : (N,) array_like
#         Input array.
#     v : (N,) array_like
#         Input array.
#     w : (N,) array_like, optional
#         The weights for each value in `u` and `v`. Default is None,
#         which gives each value a weight of 1.0
#     centered : bool, optional
#         If True, `u` and `v` will be centered. Default is True.

#     Returns
#     -------
#     correlation : double
#         The correlation distance between 1-D array `u` and `v`.

#     """
#     u = _validate_vector(u)
#     v = _validate_vector(v)
#     if w is not None:
#         w = _validate_weights(w)
#     if centered:
#         umu = np.average(u, weights=w)
#         vmu = np.average(v, weights=w)
#         u = u - umu
#         v = v - vmu
#     uv = np.average(u * v, weights=w)
#     uu = np.average(np.square(u), weights=w)
#     vv = np.average(np.square(v), weights=w)
#     dist = 1.0 - uv / np.sqrt(uu * vv)
#     # Return absolute value to avoid small negative value due to rounding
#     return np.abs(dist)


# def cosine(u, v, w=None):
#     """
#     Compute the Cosine distance between 1-D arrays.

#     The Cosine distance between `u` and `v`, is defined as

#     .. math::

#         1 - \\frac{u \\cdot v}
#                   {\\|u\\|_2 \\|v\\|_2}.

#     where :math:`u \\cdot v` is the dot product of :math:`u` and
#     :math:`v`.

#     Parameters
#     ----------
#     u : (N,) array_like
#         Input array.
#     v : (N,) array_like
#         Input array.
#     w : (N,) array_like, optional
#         The weights for each value in `u` and `v`. Default is None,
#         which gives each value a weight of 1.0

#     Returns
#     -------
#     cosine : double
#         The Cosine distance between vectors `u` and `v`.

#     Examples
#     --------
#     >>> from scipy.spatial import distance
#     >>> distance.cosine([1, 0, 0], [0, 1, 0])
#     1.0
#     >>> distance.cosine([100, 0, 0], [0, 1, 0])
#     1.0
#     >>> distance.cosine([1, 1, 0], [0, 1, 0])
#     0.29289321881345254

#     """
#     # cosine distance is also referred to as 'uncentered correlation',
#     #   or 'reflective correlation'
#     # clamp the result to 0-2
#     return max(0, min(correlation(u, v, w=w, centered=False), 2.0))


# def create_line(point1, point2, steps: int = 5):
#     """Drow line from point1 to point2, resample in the line.
#     """
#     steps = steps
#     number = int(np.ceil(np.sqrt(np.sum(np.square(point1 - point2)))/steps))
#     x = np.linspace(point1[0], point2[0], number)[1:-1]
#     y = np.linspace(point1[1], point2[1], number)[1:-1]
#     return np.array([x, y], dtype=np.int_)


# def two_coordinate_point_angle(point,
#                                target_point=np.array([1, 0]),
#                                center=np.array([0, 0])):
#     angle = angle_of_vectors(point - center,
#                              target_point - center)
#     angle = included_angle(angle)
#     return angle


# def equal_distance_resample(data, N: int = 60):
#     """
#     data: array-like
#     N: Number of points to sample
#     """
#     x = data[0]
#     y = data[1]
#     # Compute the cumulative distances
#     distances = np.sqrt(np.diff(x)**2 + np.diff(y)**2)
#     cumulative_distances = np.insert(np.cumsum(distances), 0, 0)
#     # Total curve length
#     L = cumulative_distances[-1]
#     # Create a new array of distances for interpolation
#     new_distances = np.linspace(0, L, N)
#     # Interpolate to get the new x and y values at the equally spaced distances
#     new_x = interp1d(cumulative_distances,
#                      x,
#                      kind='linear',
#                      fill_value="extrapolate")(new_distances)
#     new_y = interp1d(cumulative_distances,
#                      y,
#                      kind='linear',
#                      fill_value="extrapolate")(new_distances)
#     return np.array([new_x, new_y])


# def edges(data):
#     padded = np.pad(data,
#                     pad_width=((1,0),(0,0)),
#                     mode='constant',
#                     constant_values=False)
#     edg_0 = data & ~padded[:-1] 

#     padded = np.pad(data,
#                     pad_width=((0, 1),(0,0)),
#                     mode='constant',
#                     constant_values=False)
#     edg_1 = data & ~padded[1:] 

#     padded = np.pad(data,
#                     pad_width=((0, 0),(1,0)),
#                     mode='constant',
#                     constant_values=False)
#     edg_2 = data & ~padded[:, :-1] 

#     padded = np.pad(data,
#                     pad_width=((0, 0),(0,1)),
#                     mode='constant',
#                     constant_values=False)
#     edg_3 = data & ~padded[:, 1:] 

#     edge = edg_0 | edg_1 | edg_2 | edg_3

#     return edge


# def resample(data, n=60):
#     """resample coordiante to n
#     """
#     length = data.shape[0]
#     if data.size:
#         x = np.arange(0, length)
#         z = np.linspace(0, length, n+1)[:-1]
#         cont_x = np.interp(z, x, data[:, 0])
#         cont_y = np.interp(z, x, data[:, 1])
#         return np.stack([cont_x, cont_y], axis=1)
#     else:
#         Warning("empty data")
#         return data


# def resample_equal_distance(points, num_points):
#     # Compute the cumulative distance along the contour
#     distance = np.cumsum(np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1)))
#     distance = np.insert(distance, 0, 0)
#     # Protect against division by zero by adding a small epsilon
#     epsilon = 1e-10
#     normalized_distance = distance / (distance[-1] + epsilon)
#     # Create an array of uniformly spaced distance values
#     alpha = np.linspace(0, 1, num_points)
#     # Use interpolation to get the indices of the points in 'points' that are close to the desired distances
#     indices = np.searchsorted(normalized_distance, alpha, side='right') - 1
#     next_indices = np.clip(indices + 1, 0, len(points) - 1)
#     # Compute the weightings for the found indices
#     weights = (alpha - normalized_distance[indices]) / (normalized_distance[next_indices] - normalized_distance[indices] + epsilon)
#     # Linearly interpolate between the found points using the computed weights
#     result = (1 - weights)[:, np.newaxis] * points[indices] + weights[:, np.newaxis] * points[next_indices]
#     return result


# def angle_of_vectors(v1, v2):
#     sign_x = np.sign(v1[0])
#     if v1[0] == 0:
#         v1[0] = 1e15
#     sign_value = v2[0]*(v1[1]/v1[0]) > v2[1]
#     # cos_value = dot/(v1_norm * v2_norm)
#     cos_value = 1 - cosine(v1, v2)
#     arc_value = np.arccos(cos_value)
#     # angle_value = rotation_to_degree(arc_value)
#     if (sign_x > 0) & sign_value:
#         arc_value = np.pi*2 - arc_value
#     elif (sign_x < 0) & (~sign_value):
#         arc_value = np.pi*2 - arc_value
#     return arc_value % (np.pi*2)


# def angle_between_points_array(p1, p2):
#     dot = np.sum(p1 * p2, axis=1)
#     deter = p1[:, 0] * p2[:, 1] - p1[:, 1] * p2[:, 0]
#     anlges = np.arctan2(deter, dot)
#     anlges[anlges < 0] += 2 * np.pi
#     return anlges


# def angle_of_pints_180(p1, p2):
#     # Calculate dot product
#     dot_product = sum(a*b for a, b in zip(p1, p2))

#     # Calculate magnitudes
#     magnitude_p1 = math.sqrt(sum(a*a for a in p1))
#     magnitude_p2 = math.sqrt(sum(b*b for b in p2))

#     # Calculate cosine of the angle
#     cosine_angle = dot_product / (magnitude_p1 * magnitude_p2)

#     # Ensure the value lies between -1 and 1 (to account for potential rounding errors)
#     cosine_angle = max(-1.0, min(1.0, cosine_angle))

#     # Return the angle in radians
#     return math.acos(cosine_angle)
