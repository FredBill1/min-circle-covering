import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import art3d
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from utils.BoundedVoronoiPolygons import BoundedVoronoiPolygons
from utils.utils import unit_hypercube

DIM = 3
OUTPUT_DIR = Path("output")
SEED = 514
ELEV, AZIM = 20, 45


def get_camera_circle_points(elev: float, azim: float, samples: int = 100) -> NDArray[np.float64]:
    elev_rad, azim_rad = np.radians(elev), np.radians(azim)
    camera_vec = np.array([np.cos(elev_rad) * np.sin(azim_rad), np.cos(elev_rad) * np.cos(azim_rad), np.sin(elev_rad)])

    v1 = np.cross(camera_vec, [0, 0, 1])
    if np.allclose(v1, 0):
        v1 = np.cross(camera_vec, [1, 0, 0])
    v1 = v1 / np.linalg.norm(v1)
    v2 = np.cross(camera_vec, v1)
    v2 = v2 / np.linalg.norm(v2)

    t = np.linspace(0, 2 * np.pi, samples)
    return v1[:, np.newaxis] * np.cos(t) + v2[:, np.newaxis] * np.sin(t)


def plot_sphere(ax, center, radius, samples=20, *args, **kwargs):
    u, v = np.mgrid[0 : 2 * np.pi : samples * 2j, 0 : np.pi : samples * 1j]
    x = np.cos(u) * np.sin(v) * radius + center[0]
    y = np.sin(u) * np.sin(v) * radius + center[1]
    z = np.cos(v) * radius + center[2]
    ax.plot_surface(x, y, z, *args, **kwargs)


def visualize(X: NDArray[np.float64], limits: NDArray[np.float64]) -> None:
    limits = limits[::-1]
    X = X[:, ::-1]
    boundary = unit_hypercube(DIM) * limits
    bounded_voronoi_polygons = BoundedVoronoiPolygons(boundary)

    _, ax = plt.subplots(subplot_kw=dict(projection="3d"))
    ax.set_proj_type("persp", focal_length=0.3)
    ax.view_init(elev=ELEV, azim=AZIM)
    max_dist = 0
    circles = get_camera_circle_points(ELEV, AZIM)
    rng = np.random.default_rng(SEED)
    for xyz, center in bounded_voronoi_polygons(X):
        color = rng.random(3)
        max_dist = max(max_dist, np.square(xyz - center).sum(axis=1).max())
        ax.add_collection(art3d.Poly3DCollection(xyz[ConvexHull(xyz).simplices], color=color, edgecolor=(0,) * 4, alpha=0.1))
        plot_sphere(ax, center, 1.0, color="white", alpha=0.05, shade=True)
        ax.plot(*center, "ok", markersize=0.5, zorder=10000)
        ax.plot(*(circles.T + center).T, "-k", linewidth=0.1, zorder=10000)

    for edge in [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3, 7]]:
        ax.plot(*boundary[edge].T, "-k", linewidth=0.1)
    ax.set_aspect("equal")
    ax.set_xlim([-0.5, limits[0] + 0.5])
    ax.set_ylim([-0.5, limits[1] + 0.5])
    ax.set_zlim([-0.5, limits[2] + 0.5])

    max_dist = np.sqrt(max_dist)
    ax.title.set_text(f"num: {X.shape[0]} max_dist: {max_dist:.4f}")


def worker(file: Path) -> None:
    output_file = file.with_suffix(".png")
    with file.open("r") as f:
        data = json.load(f)
    limits = np.array(list(map(float, file.stem.split(","))), dtype=np.float64)
    x = np.array(data["x"], dtype=np.float64)
    visualize(x, limits)
    plt.savefig(output_file, dpi=300)


def main():
    files = list(OUTPUT_DIR.glob("*.json"))
    with Pool() as pool:
        pool.map(worker, files)


if __name__ == "__main__":
    main()
