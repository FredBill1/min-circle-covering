import json
from datetime import datetime
from itertools import product, repeat
from multiprocessing import Pool
from pathlib import Path
from typing import Optional

import numpy as np
import scipy.optimize
from numpy.typing import NDArray
from scipy.optimize import OptimizeResult
from tqdm import tqdm
from utils.BoundedVoronoiPolygons import BoundedVoronoiPolygons
from utils.polyhedron_sphere_intersection_volume import polyhedron_sphere_intersection_volume
from utils.utils import unit_hypercube

DIM = 3
NUM_TRIALS = 30

N = 5
OUTPUT_DIR = Path("output")


def solve(limits: NDArray[np.float64], num: int, seed: Optional[int] = None) -> tuple[NDArray[np.float64], float]:
    boundary = unit_hypercube(DIM) * limits
    bounded_voronoi_polygons = BoundedVoronoiPolygons(boundary)

    def calc(X: NDArray[np.float64]) -> tuple[float, float]:
        volume = 0
        dist2 = 0
        for vertices, center in bounded_voronoi_polygons(X):
            volume += polyhedron_sphere_intersection_volume(vertices, center, 1.0)
            dist2 = max(dist2, np.square(vertices - center).sum(axis=1).max())
        return dist2, volume

    def func(X: NDArray[np.float64]) -> float:
        X = X.reshape((-1, DIM))
        dist2, volume = calc(X)
        out_of_bound = np.clip(X - limits, 0, None).sum() + np.clip(-X, 0, None).sum()
        return out_of_bound + dist2 - volume

    target = 1.0 - limits.prod()

    def callback(intermediate_result: OptimizeResult) -> None:
        if intermediate_result.fun <= target:
            raise StopIteration

    if seed is not None:
        np.random.seed(seed)

    x0 = np.random.uniform(0, limits, (num, DIM))
    x = scipy.optimize.minimize(
        func,
        x0.flatten(),
        method="BFGS",
        callback=callback,
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
            tqdm.write(f"{datetime.now()} {dist=}")
            if dist <= 1:
                return x, dist
            if dist < min_dist:
                min_x, min_dist = x, dist
    return min_x, min_dist


def main(lim_x: float, lim_y: float, lim_z: float):
    output_file = OUTPUT_DIR / f"{lim_x},{lim_y},{lim_z}.json"
    if output_file.exists():
        return

    limits = np.array([lim_x, lim_y, lim_z], dtype=np.float64)
    l, r = 1, int(np.ceil(limits * (np.sqrt(DIM) / 2)).astype(np.int64).prod())
    ans_num, ans_x = r, None

    for dim in range(DIM):
        for t in range(1, int(limits[dim]) - 1):
            pre1, pre2 = (limits.astype(int).tolist() for _ in range(2))
            pre1[dim], pre2[dim] = t, int(limits[dim]) - t
            s_pre1, s_pre2 = (list(sorted(enumerate(pre), key=lambda t: t[1])) for pre in (pre1, pre2))

            path1, path2 = (OUTPUT_DIR / f"{','.join(map(lambda t: str(t[1]), s_pre))}.json" for s_pre in (s_pre1, s_pre2))
            if path1.exists() and path2.exists():
                with path1.open() as f1, path2.open() as f2:
                    data1, data2 = json.load(f1), json.load(f2)
                if (num := data1["num"] + data2["num"]) < r:
                    ans_num = r = num
                    rev1, rev2 = [0] * DIM, [0] * DIM
                    for i, (j, _) in enumerate(s_pre1):
                        rev1[j] = i
                    for i, (j, _) in enumerate(s_pre2):
                        rev2[j] = i
                    x1, x2 = np.array(data1["x"])[:, rev1], np.array(data2["x"])[:, rev2]
                    x2[:, dim] += t
                    ans_x = np.concatenate([x1, x2])

    for d in product(*repeat((0, -1), DIM)):
        last_file = OUTPUT_DIR / f"{','.join(map(str, limits.astype(int) + d))}.json"
        if last_file.exists():
            with last_file.open() as f:
                data = json.load(f)
            l = max(l, data["num"])

    print(f"{datetime.now()} {l=} {r=}")
    while l < r:
        num = (l + r) // 2
        x, dist = parallel_solve(limits, num)
        if dist <= 1:
            r = num
            ans_num, ans_x = num, x
            print(f"{num=}, {dist=}, {x=}")
        else:
            l = num + 1
    if ans_x is None:
        # This piece of code tries to arrange the upper bound to save time, which is added after some expirements had already been done.
        # Therefore some early results in the output folder does not match this behavior, but the number of spheres needed is unchanged.
        step = 2.0 / np.sqrt(DIM)
        ans_x = np.array(np.meshgrid(*[np.arange(step / 2, ub + step / 2, step) for ub in limits])).T.reshape(-1, DIM)
    ans_x = np.clip(ans_x, 0, limits)
    print(ans_num, ans_x)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w") as f:
        json.dump(dict(num=ans_num, x=ans_x.tolist()), f)


if __name__ == "__main__":
    tasks = [(x, y, z) for x in range(1, N + 1) for y in range(x, N + 1) for z in range(y, N + 1)]
    tasks.sort(key=lambda x: (max(x), sum(x)) + x)
    for x, y, z in tasks:
        main(x, y, z)
