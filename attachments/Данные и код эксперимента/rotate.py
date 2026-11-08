import json, sys
from bench import DATA, run
from llama_cpp import Llama

small = sys.argv[1] == "small"
llm = Llama(
    model_path="small.gguf" if small else "model.gguf",
    n_ctx=1024,
    n_threads=4,
    n_threads_batch=4,
    n_gpu_layers=0,
    verbose=False,
    seed=42,
)
ids = [llm.tokenize(t.encode(), add_bos=False)[0] for t in "ABC"]
logs = []
for shift in (1, 2):
    for project, groups in DATA.items():
        items = list(groups.items())
        reordered = dict(items[shift:] + items[:shift])
        dt, p, r = run(reordered, "qwen", llm, ids)
        logs.append(dict(project=project, shift=shift, pred=p))
        print(project, shift, round(dt, 2), flush=True)
with open("rotation-" + sys.argv[1] + ".json", "w") as f:
    json.dump(logs, f, indent=2)
