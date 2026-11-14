import json
from math import comb
from statistics import mean, median
from collections import Counter

D = json.load(open("data.json"))


def stats(gold, pred):
    labs = sorted(set(gold.values()))
    pairs = Counter((gold[k], pred[k]) for k in gold)
    rows = Counter(gold.values())
    cols = Counter(pred.values())
    f = mean(2 * pairs[l, l] / (rows[l] + cols[l]) for l in labs)
    a = sum(comb(v, 2) for v in pairs.values())
    b = sum(comb(v, 2) for v in rows.values())
    c = sum(comb(v, 2) for v in cols.values())
    exp = b * c / comb(len(gold), 2)
    ari = (a - exp) / ((b + c) / 2 - exp)
    return sum(pairs[l, l] for l in labs), f, ari


if __name__ == "__main__":
    from pathlib import Path

    modes = [
        ("TF", "tfidf", "results.json"),
        ("0.5", "qwen", "small-results.json"),
        ("1.5", "qwen", "results.json"),
        ("TF-0.5", "hybrid", "small-results.json"),
        ("TF-1.5", "hybrid", "results.json"),
    ]
    for name, mode, filename in modes:
        if not Path(filename).exists():
            continue
        rows = json.load(open(filename))
        counts = []
        for project, groups in D.items():
            subset = [r for r in rows if r["mode"] == mode and r["project"] == project]
            if not subset:
                continue
            gold = {c.split(" | ")[0]: k for k, v in groups.items() for c in v["test"]}
            assert all(s["pred"] == subset[0]["pred"] for s in subset)
            metrics = stats(gold, subset[0]["pred"])
            secs = [s["seconds"] for s in subset]
            counts.append((*metrics, median(secs), min(secs), max(secs)))
            print(
                name,
                project,
                "correct/F/ARI/med/range",
                *[round(v, 4) for v in counts[-1]]
            )
        if counts:
            print(
                "summary",
                name,
                sum(x[0] for x in counts),
                mean(x[1] for x in counts),
                mean(x[2] for x in counts),
                sum(x[3] for x in counts),
            )
    for size, extra, base in [
        ("1.5", "rotation-big.json", "results.json"),
        ("0.5", "rotation-small.json", "small-results.json"),
    ]:
        if not Path(extra).exists():
            continue
        base = json.load(open(base))
        allrots = json.load(open(extra))
        sumrot = [0, 0, 0]
        stable = 0
        pos = Counter()
        for p, g in D.items():
            gold = {c.split(" | ")[0]: k for k, v in g.items() for c in v["test"]}
            placed = [
                next(s for s in base if s["mode"] == "qwen" and s["project"] == p)[
                    "pred"
                ]
            ] + [r["pred"] for r in allrots if r["project"] == p]
            cts = [stats(gold, pred)[0] for pred in placed]
            sumrot = [sum(x) for x in zip(sumrot, cts)]
            stable += sum(len({pr[k] for pr in placed}) == 1 for k in gold)
            for shift, pred in enumerate(placed):
                pos.update((list(g).index(v) - shift) % 3 for v in pred.values())
            print(size, p, "rotation", cts)
        print(size, "sum", sumrot, "stable", stable, "positions", pos)
