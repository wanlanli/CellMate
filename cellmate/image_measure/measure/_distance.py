import numpy as np
from sklearn.neighbors import NearestNeighbors


def find_nearest_points(x, y):
    """
    x: feature point list
    y: feature point list
    """
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(y)
    distances, index = nbrs.kneighbors(x)
    dis = np.min(distances)
    ind = np.argmin(distances)
    return dis, ind, index[ind][0]
