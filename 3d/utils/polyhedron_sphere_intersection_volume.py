from typing import Any

import numpy as np
import overlap
from numpy.typing import NDArray
from scipy.spatial import ConvexHull


def polyhedron_sphere_intersection_volume(
    vertices: NDArray[np.floating[Any]], center: NDArray[np.floating[Any]], radius: float
) -> NDArray[np.floating[Any]]:
    ans = 0.0
    m = vertices.mean(axis=0, keepdims=True)
    for simplex in ConvexHull(vertices).simplices:
        v = vertices[simplex]
        tet = overlap.Tetrahedron(np.r_[v, m] if np.cross(v[1] - v[0], v[2] - v[0]).dot(m[0] - v[0]) > 0 else np.r_[m, v])
        sphere = overlap.Sphere(center, radius)
        ans += overlap.overlap(sphere, tet)
    return ans
