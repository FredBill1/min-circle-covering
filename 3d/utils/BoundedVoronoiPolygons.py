from collections.abc import Generator
from typing import Optional

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog
from scipy.spatial import ConvexHull, HalfspaceIntersection, Voronoi


def halfspace_intersection(halfspaces: NDArray[np.float64]) -> Optional[NDArray[np.float64]]:
    norm_vector = np.reshape(np.linalg.norm(halfspaces[:, :-1], axis=1), (halfspaces.shape[0], 1))
    c = np.zeros((halfspaces.shape[1],))
    c[-1] = -1
    A = np.hstack((halfspaces[:, :-1], norm_vector))
    b = -halfspaces[:, -1:]
    res = linprog(c, A_ub=A, b_ub=b, bounds=(None, None))
    x, y = res.x[:-1], res.x[-1]
    return HalfspaceIntersection(halfspaces, x).intersections if y > 0 else None


class BoundedVoronoiPolygons:
    def __init__(self, boundary: NDArray[np.float64]) -> None:
        hull = ConvexHull(boundary)
        self._boundary_halfspaces = hull.equations
        centroid = np.mean(boundary, axis=0)
        self._append = (boundary - centroid) * 4 + centroid

    def __call__(self, X: NDArray[np.float64]) -> Generator[tuple[NDArray[np.float64], NDArray[np.float64]], None, None]:
        vor = Voronoi(np.r_[X, self._append])  # add additional points outside the boundary, to make sure the voronoi diagram can always be computed
        for r, center in zip(vor.point_region, X):
            region = vor.regions[r]
            if -1 in region:  # infinite region
                continue
            halfspaces = np.r_[self._boundary_halfspaces, ConvexHull(vor.vertices[region]).equations]
            if (intersection := halfspace_intersection(halfspaces)) is None:  # no intersection
                continue
            yield intersection, center
