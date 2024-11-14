import json
from multiprocessing import Pool
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from scipy.spatial import ConvexHull
from utils.BoundedVoronoiPolygons import BoundedVoronoiPolygons
from utils.utils import unit_hypercube

DIM = 2
OUTPUT_DIR = Path("output")
SEED = 514


def visualize(X: NDArray[np.float64], limits: NDArray[np.float64]) -> None:
    if limits[0] < limits[1]:
        limits = limits[::-1]
        X = X[:, ::-1]
    boundary = unit_hypercube(DIM) * limits
    bounded_voronoi_polygons = BoundedVoronoiPolygons(boundary)

    _, ax = plt.subplots(subplot_kw=dict(projection="3d") if DIM == 3 else {})
    max_dist = 0
    rng = np.random.default_rng(SEED)
    for xy, center in bounded_voronoi_polygons(X):
        color = rng.random(3)
        max_dist = max(max_dist, np.square(xy - center).sum(axis=1).max())
        ax.fill(*xy[ConvexHull(xy).vertices].T, color=color, alpha=0.5)
        ax.plot(*center, "ok", markersize=2, zorder=10000)
        ax.add_artist(plt.Circle(center, 1, color="k", linewidth=0.5, fill=False, zorder=10000))
    ax.set_aspect("equal")

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
