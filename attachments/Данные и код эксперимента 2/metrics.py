from analysis import D, stats
from collections import Counter
from statistics import mean, median
import json
from pathlib import Path

groups = json.load(open("grouping-results.json"))
for mode in ("cluster", "0.5", "1.5"):
    print("MODE", mode)
    s = []
    for proj, ps in D.items():
        gold = {c.split(" | ")[0]: k for k, v in ps.items() for c in v["test"]}
        arr = [r for r in groups if r["mode"] == mode and r["project"] == proj]
        if not arr:
            continue
        assert all(r["pred"] == arr[0]["pred"] for r in arr)
        ari = stats(gold, arr[0]["pred"])[2]
        times = [r["seconds"] for r in arr]
        s.append([ari, median(times)])
        print(
            proj,
            "ARI",
            round(ari, 4),
            "time",
            round(median(times), 4),
            "range",
            min(times),
            max(times),
            "sizes",
            list(Counter(arr[0]["pred"].values()).values()),
        )
    if s:
        print("summary", mean(a for a, t in s), sum(t for a, t in s))
