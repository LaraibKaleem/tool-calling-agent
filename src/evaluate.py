#!/usr/bin/env python3
import sys, argparse, pathlib, json, time, importlib.util
sys.path.insert(0, ".")
sys.path.insert(0, "starter")

def parse_output(raw):
    import re
    m = re.search(r"<tool_call>(.*?)</tool_call>", raw, re.DOTALL)
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
            ok = abs(ef - gf) / abs(ef) <= 0.01 if ef != 0 else gf == 0
            if not ok:
                all_correct = False; break
        except:
            if str(ev).strip().upper() != str(gv).strip().upper():
                all_correct = False; break
    return 1.0 if all_correct else 0.5

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--test",      default="starter/public_test.jsonl")
    ap.add_argument("--inference", default="inference.py")
    ap.add_argument("--slice",     default=None)
    ap.add_argument("--verbose",   action="store_true")
    ap.add_argument("--latency",   action="store_true")
    args = ap.parse_args()

    spec = importlib.util.spec_from_file_location("inference", args.inference)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    examples = [json.loads(l) for l in pathlib.Path(args.test).read_text().splitlines() if l.strip()]
    if args.slice:
        examples = [e for e in examples if e.get("slice","") == args.slice]

    print(f"Evaluating {len(examples)} examples")
    print("-" * 50)

    all_scores = []
    latencies  = []

    for ex in examples:
        t0    = time.perf_counter()
        raw   = mod.run(ex["prompt"], ex.get("history",[]))
        lat   = (time.perf_counter() - t0) * 1000
        score = score_example(ex["expected"], raw)
        all_scores.append(score)
        latencies.append(lat)

        icon = "✅" if score==1.0 else ("⚠️" if score==0.5 else ("🚫" if score==-0.5 else "❌"))
        print(f"{icon} [{ex.get('id','?'):6s}] score={score:+.1f} lat={lat:5.1f}ms")
        if args.verbose:
            print(f"     prompt: {ex['prompt'][:70]}")
            print(f"     output: {raw[:70]}")

    print("\n" + "="*50)
    print(f"Mean score : {sum(all_scores)/len(all_scores):.3f}")
    print(f"Perfect ✅ : {sum(1 for s in all_scores if s==1.0)}/{len(all_scores)}")
    print(f"Partial ⚠️ : {sum(1 for s in all_scores if s==0.5)}/{len(all_scores)}")
    print(f"Zero    ❌ : {sum(1 for s in all_scores if s==0.0)}/{len(all_scores)}")
    print(f"Penalty 🚫 : {sum(1 for s in all_scores if s==-0.5)}/{len(all_scores)}")
    if latencies:
        avg = sum(latencies)/len(latencies)
        print(f"Avg latency: {avg:.1f}ms  Gate: {'✅ PASS' if avg<=200 else '❌ FAIL'}")
    print("="*50)

if __name__ == "__main__":
    main()