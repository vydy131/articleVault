from collections import Counter
from math import sqrt, log
from time import perf_counter
import json
import re
from samples import hashname

DATA = json.load(open("data.json"))
MODES = ["tfidf", "qwen", "hybrid"]
SYSTEM = "Assign the Python class to one of the candidate software modules. Use the module descriptions and labeled examples. Reply with a single letter A, B or C."


def words(s):
    s = re.sub("([A-Z]+)([A-Z][a-z])", r"\1 \2", s)
    s = re.sub("([a-z0-9])([A-Z])", r"\1 \2", s)
    return Counter(re.findall("[a-z0-9]+", s.lower()))


def options(d):
    return [v["label"] + " ; " + " ; ".join(v["anchors"]) for v in d.values()]


def lexical(descriptions):
    profiles = [words(d) for d in descriptions]
    df = Counter()
    for p in profiles:
        df.update(p.keys())
    idf = {t: 1 + log(4 / (f + 1)) for t, f in df.items()}

    def vec(p):
        v = {t: n * idf[t] for t, n in p.items() if t in idf}
        norm = sqrt(sum(val * val for val in v.values())) or 1
        return {t: val / norm for t, val in v.items()}

    pvec = [vec(p) for p in profiles]

    def predict(card):
        v = vec(words(card))
        scores = [sum(val * p.get(t, 0) for t, val in v.items()) for p in pvec]
        best = max(scores)
        return [i for i, s in enumerate(scores) if abs(s - best) <= 1e-12]

    return predict


def run(d, mode, llm, token_ids):
    start = perf_counter()
    des = options(d)
    classify = lexical(des) if mode != "qwen" else None
    prefix = (
        "<|im_start|>system\n"
        + SYSTEM
        + "<|im_end|>\n<|im_start|>user\n"
        + "\n".join(f"{chr(65+i)}. {desc}" for i, desc in enumerate(des))
        + "\nClass: "
    )
    prefix_toks = None
    pred = {}
    routed = 0
    cards = sorted(
        (card for v in d.values() for card in v["test"]),
        key=lambda c: hashname(c.split(" | ")[0]),
    )
    for card in cards:
        ties = classify(card) if classify else None
        if mode == "tfidf" or (mode == "hybrid" and len(ties) == 1):
            label = ties[0]
        else:
            if prefix_toks is None:
                prefix_toks = llm.tokenize(prefix.encode(), add_bos=False, special=True)
                llm.reset()
                llm.eval(prefix_toks)
            suffix = card + "<|im_end|>\n<|im_start|>assistant\n"
            llm.n_tokens = len(prefix_toks)
            llm.eval(llm.tokenize(suffix.encode(), add_bos=False, special=True))
            logits = llm._ctx.get_logits_ith(-1)
            label = max(range(3), key=lambda i: logits[token_ids[i]])
            routed += 1
        pred[card.split(" | ")[0]] = list(d)[label]
    return perf_counter() - start, pred, routed


if __name__ == "__main__":
    import argparse
    from llama_cpp import Llama

    parser = argparse.ArgumentParser(
        description="Assign held-out classes to known modules."
    )
    parser.add_argument("--size", choices=["0.5", "1.5"], default="1.5")
    size = parser.parse_args().size
    model_file = "small.gguf" if size == "0.5" else "model.gguf"
    results_file = "small-results.json" if size == "0.5" else "results.json"
    llm = Llama(
        model_path=model_file,
        n_ctx=1024,
        n_threads=4,
        n_threads_batch=4,
        n_gpu_layers=0,
        verbose=False,
        seed=42,
    )
    token_ids = []
    for ch in "ABC":
        tt = llm.tokenize(ch.encode(), add_bos=False)
        assert len(tt) == 1
        token_ids.extend(tt)
    llm.eval(llm.tokenize(b"Warm up.", add_bos=False))
    llm.reset()
    logs = []
    projects = list(DATA)
    for r in range(3):
        for ix in range(3):
            name = projects[(r + ix) % 3]
            for mi in range(3):
                mode = MODES[(mi + r + ix) % 3]
                dt, p, rt = run(DATA[name], mode, llm, token_ids)
                row = {
                    "repeat": r,
                    "project": name,
                    "mode": mode,
                    "seconds": dt,
                    "pred": p,
                    "calls": rt,
                }
                print(r, name, mode, round(dt, 4), rt, flush=True)
                logs.append(row)
                with open(results_file, "w") as f:
                    json.dump(logs, f, indent=2)
