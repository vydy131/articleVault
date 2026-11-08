import json, sys
from math import sqrt, log
from time import perf_counter
from bench import DATA, words, hashname

SYS = "Group Python classes by business responsibility into exactly three cohesive software modules. Module names are unknown. Reply with one space-separated line containing one letter per input class in the same order. Use A for the first group, B for the next new group and C for the third."


def cluster(cards):
    tok = [words(c) for c in cards]

    def vector(doc):
        vals = {
            t: f * (1 + log((1 + len(cards)) / (1 + sum(t in d for d in tok))))
            for t, f in doc.items()
        }
        denom = sqrt(sum(x * x for x in vals.values())) or 1
        return {t: v / denom for t, v in vals.items()}

    vs = list(map(vector, tok))
    distances = {
        (i, j): 1 - sum(x * vs[j].get(t, 0) for t, x in vs[i].items())
        for i in range(len(vs))
        for j in range(i + 1, len(vs))
    }
    groups = [(i,) for i in range(len(vs))]
    while len(groups) > 3:

        def key(pair):
            a, b = pair
            dist = sum(distances[tuple(sorted((i, j)))] for i in a for j in b) / (
                len(a) * len(b)
            )
            return dist, a, b

        a, b = min(
            ((a, b) for i, a in enumerate(groups) for b in groups[i + 1 :]), key=key
        )
        groups = sorted(
            [g for g in groups if g != a and g != b] + [tuple(sorted(a + b))]
        )
    return {
        cards[i].split(" | ")[0]: chr(65 + k) for k, g in enumerate(groups) for i in g
    }


def neural(cards, llm):
    listing = "\n".join(f"{i}. {c}" for i, c in enumerate(cards, 1))
    prompt = (
        "<|im_start|>system\n"
        + SYS
        + "<|im_end|>\n<|im_start|>user\n"
        + listing
        + "<|im_end|>\n<|im_start|>assistant\n"
    )
    llm.reset()
    llm.eval(llm.tokenize(prompt.encode(), add_bos=False, special=True))
    used = 0
    pred = {}
    for i, card in enumerate(cards):
        letters = "ABC"[: min(used + 1, 3)]
        if len(cards) - i == 3 - used:
            letters = "ABC"[used : used + 1]
        ids = {
            ch: llm.tokenize(((" " if i else "") + ch).encode(), add_bos=False)
            for ch in letters
        }
        assert all(len(v) == 1 for v in ids.values())
        logits = llm._ctx.get_logits_ith(-1)
        ch = max(letters, key=lambda c: logits[ids[c][0]])
        llm.eval(ids[ch])
        pred[card.split(" | ")[0]] = ch
        used = max(used, ord(ch) - 64)
    assert set(pred.values()) == set("ABC")
    return pred


if __name__ == "__main__":
    from llama_cpp import Llama

    rows = []
    for mode, model in [
        ("cluster", None),
        ("0.5", "small.gguf"),
        ("1.5", "model.gguf"),
    ]:
        llm = None
        if model:
            llm = Llama(
                model_path=model,
                n_ctx=1024,
                n_threads=4,
                n_threads_batch=4,
                n_gpu_layers=0,
                verbose=False,
                seed=42,
            )
            llm.eval(llm.tokenize(b"Warm up.", add_bos=False))
            llm.reset()
        for rep in range(3):
            names = list(DATA)
            for ix in range(3):
                project = names[(rep + ix) % 3]
                start = perf_counter()
                cards = sorted(
                    (c for v in DATA[project].values() for c in v["test"]),
                    key=lambda c: hashname(c.split(" | ")[0]),
                )
                pred = neural(cards, llm) if llm else cluster(cards)
                seconds = perf_counter() - start
                rows.append(
                    dict(
                        project=project,
                        mode=mode,
                        repeat=rep,
                        pred=pred,
                        seconds=seconds,
                    )
                )
                print(mode, rep, project, round(seconds, 4), flush=True)
                with open("grouping-results.json", "w") as f:
                    json.dump(rows, f, indent=2)
        if llm:
            llm.close()
