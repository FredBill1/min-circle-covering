import json
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from tqdm import tqdm
from utils.BoundedVoronoiPolygons import BoundedVoronoiPolygons
from utils.polygon_circle_intersection import polygon_circle_intersection
from utils.utils import unit_hypercube

DIM = 2
NUM_TRIALS = 30

N = 9
OUTPUT_DIR = Path("output")


def solve(limits: NDArray[np.float64], num: int, seed: Optional[int] = None) -> tuple[NDArray[np.float64], float]:
    boundary = unit_hypercube(DIM) * limits
    bounded_voronoi_polygons = BoundedVoronoiPolygons(boundary)

    def calc(X: NDArray[np.float64]) -> tuple[float, float]:
        area = 0
        dist2 = 0
        for xy, center in bounded_voronoi_polygons(X):
            xy = xy - center
            area += polygon_circle_intersection(1.0, xy[::-1].T)[0]
            dist2 = max(dist2, np.square(xy).sum(axis=1).max())
        return dist2, area

    def func(X: NDArray[np.float64]) -> float:
        X = X.reshape((-1, DIM))
        dist2, area = calc(X)
        out_of_bound = np.clip(X - limits, 0, None).sum() + np.clip(-X, 0, None).sum()
        return out_of_bound + dist2 - area

    if seed is not None:
        np.random.seed(seed)

    x0 = np.random.uniform(0, limits, (num, DIM))
    x = scipy.optimize.minimize(
        func,
        x0.flatten(),
        method="BFGS",
    ).x.reshape((-1, DIM))

    dist2, _ = calc(x)
    return x, np.sqrt(dist2)


def worker(args: tuple[NDArray[np.float64], int]) -> tuple[NDArray[np.float64], float]:
    return solve(*args)


def parallel_solve(limits: NDArray[np.float64], num: int) -> tuple[NDArray[np.float64], float]:
    min_x, min_dist = None, np.inf
    tasks = [(limits, num, seed) for seed in np.random.randint(0, 2**31, NUM_TRIALS)]
    with Pool() as pool:
        for x, dist in tqdm(pool.imap_unordered(worker, tasks), total=NUM_TRIALS, desc=f"{num=} limits={limits.tolist()}"):
            tqdm.write(f"{dist=}")
            if dist <= 1:
                return x, dist
            if dist < min_dist:
                min_x, min_dist = x, dist
    return min_x, min_dist


def main(lim_x: float, lim_y: float):
    output_file = OUTPUT_DIR / f"{lim_x},{lim_y}.json"
    if output_file.exists():
        return

    limits = np.array([lim_x, lim_y], dtype=np.float64)
    l, r = 1, int(np.ceil(limits / np.sqrt(2)).astype(np.int64).prod()) + 1
    for dx, dy in ((-1, 0), (0, -1), (-1, -1)):
        last_file = OUTPUT_DIR / f"{lim_x+dx},{lim_y+dy}.json"
        if last_file.exists():
            with last_file.open() as f:
                data = json.load(f)
            l = max(l, data["num"])
    print(f"{l=} {r=}")

    ans_num, ans_x = [None] * 2
    while l < r:
        num = (l + r) // 2
        x, dist = parallel_solve(limits, num)
        if dist <= 1:
            r = num
            ans_num, ans_x = num, x
            print(f"{num=}, {dist=}, {x=}")
        else:
            l = num + 1
    print(ans_num, ans_x)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(dict(num=ans_num, x=ans_x.tolist()), f)


if __name__ == "__main__":
    tasks = [(w, h) for w in range(1, N + 1) for h in range(w, N + 1)]
    tasks.sort(key=lambda x: (max(x), sum(x)))
    for w, h in tasks:
        main(w, h)
