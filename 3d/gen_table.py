import json
from itertools import product
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("output")
N = 5


def read_num(path: Path) -> tuple[tuple[int, int, int], int]:
    with path.open() as f:
        data = json.load(f)
    limits = tuple(map(int, path.stem.split(",")))
    num = data["num"]
    return limits, num


def main():
    files = OUTPUT_DIR.glob("*.json")
    with ThreadPool() as pool:
        mp = dict(pool.map(read_num, files))
    results = []
    for a, b, c in product(range(1, N + 1), repeat=3):
        num = mp[tuple(sorted((a, b, c)))]
        results.append((a, b, c, num))
    df = pd.DataFrame(results, columns=["x", "y", "z", "num"])
    df.to_csv(OUTPUT_DIR / "full.csv", index=False)
    for x in range(1, N + 1):
        cur = df[df["x"] == x].pivot(index="y", columns="z", values="num")
        print(f"{x=}\n{cur}\n")
        cur.to_csv(OUTPUT_DIR / f"{x}.csv")
    print("Done")


if __name__ == "__main__":
    main()
