import numpy as np
from sklearn.neighbors import NearestNeighbors

class CoordTree():
    def __init__(self, coord: None, top_n: int = 2) -> None:
        self.coord = coord
        if self.coord is None:
            self.tree = None
        else:
            self.tree = NearestNeighbors(n_neighbors=top_n, algorithm='ball_tree').fit(coord)

    def nearest(self, x):
        """
        Finds the nearest point in coord set to a given point or set of points x.

        Parameters:
        -----------
        x : array-like
            A 2D array or list of points where each point is represented by its (x, y) coordinates.
            Shape of x should be (m, 2), where m is the number of points to find the nearest neighbors for.

        Returns:
        --------
        dis : float
            The minimum distance between the points in x and their nearest neighbors in the set stored in self.tree.

        ind : int
            The index of the point in x that has the nearest neighbor in the set stored in self.tree.

        nearest_point_index : int
            The index of the nearest neighbor in the set stored in self.tree corresponding to the point in x.
        """
        if self.coord is None:
            return None
        distances, index = self.tree.kneighbors(x)
        dis = np.min(distances)
        ind = np.argmin(distances)
        return dis, ind, index[ind][0]

    def topn(self, x, top_n=1):
        """
        Finds the top-n nearest neighbors for each point in x.
        Parameters:
        -----------
        x : array-like
            A 2D array or list of points where each point is represented by its (x, y) coordinates.

        top_n : int, optional
            The number of nearest neighbors to return. Default is 1.

        Returns:
        --------
        distances : numpy.ndarray
            A 2D array where each row contains the distances to the top-n nearest neighbors for each point in x.

        index : numpy.ndarray
            A 2D array where each row contains the indices of the top-n nearest neighbors in the stored set for each point in x.
        """
        if self.coord is None:
            return None
        distances, index = self.tree.kneighbors(x)
        return distances[:, :top_n], index[:, :top_n]
