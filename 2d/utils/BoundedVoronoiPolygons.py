from collections.abc import Generator

import numpy as np
import shapely
from numpy.typing import NDArray
from scipy.spatial import ConvexHull, Voronoi
from shapely.geometry import Polygon
from shapely.geometry.base import BaseGeometry


class BoundedVoronoiPolygons:
    def __init__(self, boundary: NDArray[np.float64]) -> None:
        boundary = boundary[ConvexHull(boundary).vertices]
        self._boundary = Polygon(boundary)
        shapely.prepare(self._boundary)
        centroid = np.mean(boundary, axis=0)
        self._append = (boundary - centroid) * 4 + centroid

    def __call__(self, X: NDArray[np.float64]) -> Generator[tuple[NDArray[np.float64], NDArray[np.float64]], None, None]:
        vor = Voronoi(np.r_[X, self._append])  # add additional points outside the boundary, to make sure the voronoi diagram can always be computed
        for r, center in zip(vor.point_region, X):
            region = vor.regions[r]
            if -1 in region:  # infinite region
                continue
            polygon = Polygon(vor.vertices[region])
            intersection: BaseGeometry = polygon.intersection(self._boundary)
            if intersection.is_empty or not isinstance(intersection, Polygon):
                continue
            yield np.c_[intersection.exterior.coords.xy], center
