import ast
import hashlib
import json
import re
import tarfile

DOMAINS = {
    "saleor": {
        "discount": "discounts and promotions",
        "order": "orders and fulfillment",
        "product": "product catalog",
    },
    "netbox": {
        "circuits": "telecom circuits and providers",
        "dcim": "data center infrastructure management",
        "ipam": "IP address management",
    },
    "erpnext": {
        "accounts": "accounts and finance",
        "selling": "sales and customers",
        "stock": "stock and inventory",
    },
}


def hashname(name):
    return hashlib.sha256(name.encode()).digest()


def describe(node):
    members = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.append(item.name)
        elif isinstance(item, (ast.AnnAssign, ast.Assign)):
            targets = [item.target] if isinstance(item, ast.AnnAssign) else item.targets
            members.extend(t.id for t in targets if isinstance(t, ast.Name))
    public = list(dict.fromkeys(m for m in members if not m.startswith("_")))
    return node.name + ((" | " + ", ".join(public[:5])) if public else "")


def dataset():
    data = {}
    for project, domains in DOMAINS.items():
        pool = {key: {} for key in domains}
        with tarfile.open(project + ".tar.gz") as src:
            for entry in src:
                if project == "erpnext":
                    m = re.search(
                        r"/erpnext/([^/]+)/doctype/([^/]+)/\2\.py$", entry.name
                    )
                else:
                    m = re.search(
                        "/" + project + r"/([^/]+)/(?:models\.py|models/[^/]+\.py)$",
                        entry.name,
                    )
                if not m or m[1] not in domains:
                    continue
                for node in ast.parse(src.extractfile(entry).read()).body:
                    if not isinstance(node, ast.ClassDef) or node.name.lower().endswith(
                        ("queryset", "mixin")
                    ):
                        continue
                    if (
                        project == "erpnext"
                        and node.name.lower() != m[2].replace("_", "").lower()
                    ):
                        continue
                    assert node.name not in pool[m[1]]
                    pool[m[1]][node.name] = describe(node)
        unique = [n for group in pool.values() for n in group]
        data[project] = {}
        for domain, classes in pool.items():
            chosen = [
                classes[n]
                for n in sorted(classes, key=hashname)
                if unique.count(n) == 1
            ]
            assert len(chosen) >= 7, (project, domain, len(chosen))
            data[project][domain] = {
                "label": domains[domain],
                "n": len(chosen),
                "anchors": chosen[:2],
                "test": chosen[2:7],
            }
    return data


if __name__ == "__main__":
    data = dataset()
    with open("data.json", "w") as f:
        json.dump(data, f, indent=2)
    for p, d in data.items():
        print(p, [(k, v["n"], v["anchors"], v["test"]) for k, v in d.items()])
