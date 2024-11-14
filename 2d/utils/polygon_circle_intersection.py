"""
Translated to Python from https://uk.mathworks.com/matlabcentral/fileexchange/126645-intersection-of-polygon-and-circle

% polygonCircleIntersection - calculates area of intersection of circle
% of radius R centered in origin with a 2D polygon with vertices in P, and
% provides shape of intersection
% 
% Inputs:
%   R : radius of circle [scalar]
%   P : 2 X nV matrix of polygon vertices. Each column is a vertex
% Vertices must be in counter-clockwise order or things will go wrong
%
% Outputs:
%   A : area of intersection
%   pointList : list of points on the perimiter of the intersection
%   isArc : array of logicals. isArc(i) is true if the segment between 
%   i-1 and i is an arc.
%   AP : the area of the polygon
%
% Used for FOV analysis
%
% See also: intersect, polyshape, triangulation
%
% Author: Simão da Graça Marto
% e-mail: simao.marto@gmail.com
% Date: 22/03/2023
"""

import numpy as np
from numpy import floating
from numpy.typing import NDArray
from shapely.geometry import Point, Polygon


def polygon_circle_intersection(R: float, P: NDArray[floating]) -> tuple[float, NDArray[floating], list[bool], float]:
    """
    Calculates area of intersection of circle of radius R centered in origin
    with a 2D polygon with vertices in P, and provides shape of intersection.

    Parameters
    ----------
    R (float): radius of circle
    P (numpy.ndarray): 2 x nV matrix of polygon vertices. Each column is a vertex.
                       Vertices must be in counter-clockwise order.

    Returns
    -------
    tuple: (A, point_list, is_arc, AP)
        A (float): area of intersection
        point_list (numpy.ndarray): list of points on the perimeter of the intersection
        is_arc (list): array of booleans. is_arc[i] is True if the segment between i-1 and i is an arc.
        AP (float): the area of the polygon
    """
    nV = P.shape[1]
    AP = 0
    for i in range(nV):
        AP += shoelace(P[:, i], P[:, (i + 1) % nV])

    # Sanity check: order of points
    if AP < 0:
        raise ValueError("Polygon must be in counter-clockwise order")

    nT2 = np.sum(P**2, axis=0)
    is_outside = nT2 > R**2

    # 1st edge case: all points inside, so polygon fully inside
    if not np.any(is_outside):
        return AP, P, [False] * nV, AP

    # Cycle vertices to start from an outside vertex
    shift_k = -np.argmax(is_outside)
    P = np.roll(P, shift_k, axis=1)
    is_outside = np.roll(is_outside, shift_k)

    # Compute intersection
    point_list = []
    is_arc = []
    outside_circle = True

    for i in range(nV):
        x0 = P[:, i]
        d = P[:, (i + 1) % nV] - x0
        xI, on_circle = segment_circle_intersection(x0, d, R)

        if len(xI) > 0:
            point_list.extend(xI)
            is_arc.append(outside_circle)
            is_arc.append(False)
            outside_circle = on_circle[1]

    # Edge cases
    if not point_list and np.all(is_outside):
        if Polygon(P.T).contains(Point(0, 0)):  # 2nd edge case: circle fully inside triangle
            return np.pi * R**2, np.array([[1], [0]]), [], AP
        else:  # 3rd edge case: triangle fully outside circle
            return 0, np.array([]), [], AP

    # Compute area as a sum of triangles (shoelace) and arcs
    A = 0
    nI = len(point_list)
    point_list = np.array(point_list).T

    for i in range(nI):
        i_prev = (i - 1) % nI
        x_prev = point_list[:, i_prev]
        xi = point_list[:, i]

        if is_arc[i]:
            th_prev = np.arctan2(x_prev[1], x_prev[0])
            thi = np.arctan2(xi[1], xi[0])
            dth = (thi - th_prev + 2 * np.pi) % (2 * np.pi)
            if dth == 2 * np.pi:
                dth = 0
            A += dth * R**2 / 2
        else:
            A += shoelace(x_prev, xi)

    AC = np.pi * R**2
    if A > AC or A > AP:
        # raise ValueError("Invalid area")
        print(f"Warning: area of intersection is larger than circle or polygon area: {A=}, {AC=}, {AP=}, {R=}, {P=}")
        A = min(AC, AP)

    return A, point_list, is_arc, AP


def segment_circle_intersection(x0, d, R):
    """
    Calculate intersection points between a line segment and a circle.

    Args:
    x0 (numpy.ndarray): Start point of the segment
    d (numpy.ndarray): Direction vector of the segment
    R (float): Radius of the circle

    Returns:
    tuple: (xI, on_circle)
        xI (list): List of intersection points
        on_circle (list): List indicating if start/end of segment is on the circle
    """
    a = np.sum(d**2)
    b = 2 * np.dot(x0, d)
    c = np.sum(x0**2) - R**2
    D = b**2 - 4 * a * c

    if D <= 0:
        return [], []

    ll = (-b + np.array([-1, 1]) * np.sqrt(D)) / (2 * a)

    if ll[1] < 0 or ll[0] > 1:
        return [], []

    ll = np.clip(ll, 0, 1)
    xI = [x0 + l * d for l in ll]
    on_circle = [ll[0] > 0, ll[1] < 1]

    return xI, on_circle


def shoelace(p1, p2):
    """
    Shoelace formula for area of a straight segment with edges connected to origin.

    Args:
    p1 (numpy.ndarray): First point
    p2 (numpy.ndarray): Second point

    Returns:
    float: Area (can be negative if area is meant to be subtracted)
    """
    return (p1[0] * p2[1] - p1[1] * p2[0]) / 2
