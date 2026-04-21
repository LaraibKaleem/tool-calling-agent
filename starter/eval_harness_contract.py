import json, re, importlib.util, pathlib

_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def parse_output(raw):
    m = _TAG_RE.search(raw)
    if not m:
        return None
    try:
        return json.loads(m.group(1).strip())
    except:
        return {"__malformed__": True}

def score_example(expected, raw_output):
    parsed = parse_output(raw_output)
    if expected["type"] == "refusal":
        return 1.0 if parsed is None else -0.5
    if parsed is None:
        return 0.0
    if parsed.get("__malformed__"):
        return 0.0
    if parsed.get("tool") != expected["tool"]:
        return 0.0
    exp_args = expected.get("args", {})
    got_args = parsed.get("args", {})
    all_correct = True
    for k, ev in exp_args.items():
        gv = got_args.get(k)
        if gv is None:
            all_correct = False; break
        try:
            ef, gf = float(ev), float(gv)
            ok = abs(ef-gf)/abs(ef) <= 0.01 if ef != 0 else gf == 0
            if not ok:
                all_correct = False; break
        except:
            if str(ev).strip().upper() != str(gv).strip().upper():
                all_correct = False; break
    return 1.0 if all_correct else 0.5

def check_no_network_imports(path="inference.py"):
    import ast
    src  = pathlib.Path(path).read_text()
    tree = ast.parse(src)
    banned = {"requests","urllib","http","socket","httpx","aiohttp"}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) \
                    else ([node.module] if node.module else [])
            for n in names:
                if n and n.split(".")[0] in banned:
                    return False
    return True

def evaluate(test_path, inference_path="inference.py"):
    assert check_no_network_imports(inference_path), "FAIL: network import in inference.py"
    spec = importlib.util.spec_from_file_location("inference", inference_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    examples = [json.loads(l) for l in pathlib.Path(test_path).read_text().splitlines() if l.strip()]
    scores   = []
    for ex in examples:
        raw = mod.run(ex["prompt"], ex.get("history",[]))
        s   = score_example(ex["expected"], raw)
        scores.append(s)
        print(f"  [{ex.get('id','?'):20s}] score={s:+.1f}")
    mean = sum(scores)/len(scores)
    print(f"\nMean score: {mean:.3f}")
    return mean

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",      default="starter/public_test.jsonl")
    ap.add_argument("--inference", default="inference.py")
    args = ap.parse_args()
    evaluate(args.test, args.inference)