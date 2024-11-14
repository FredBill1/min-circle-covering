import json
from multiprocessing.pool import ThreadPool
from pathlib import Path

import pandas as pd

OUTPUT_DIR = Path("output")
N = 9


def worker(file: Path) -> tuple[int, int, int]:
    w, h = map(int, file.stem.split(","))
    with file.open() as f:
        data = json.load(f)
    num = data["num"]
    return w, h, num


def main():
    files = list(OUTPUT_DIR.glob("*.json"))
    with ThreadPool() as pool:
        results = pool.map(worker, files)
    results = [(w, h, num) for w, h, num in results if max(w, h) <= N]
    results.extend([(h, w, num) for w, h, num in results if w != h])
    df = pd.DataFrame(results, columns=["width", "height", "num"])
    df = df.pivot(index="height", columns="width", values="num")
    print(df)
    df.to_csv(OUTPUT_DIR / "output.csv")


if __name__ == "__main__":
    main()
