#!/usr/bin/env python3
import argparse, pathlib, subprocess, sys, json, hashlib

LLAMA_CPP_URL = "https://github.com/ggerganov/llama.cpp.git"

def run(cmd):
    print(f"  $ {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

def get_size_mb(path):
    return path.stat().st_size / 1e6

def sha256(path):
    h = hashlib.sha256()
    with open(path,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged",       default="artifacts/merged_model")
    ap.add_argument("--out-dir",      default="artifacts/quantized")
    ap.add_argument("--quant",        nargs="+", default=["Q4_K_M"])
    ap.add_argument("--llama-cpp-dir",default="llama.cpp")
    return ap.parse_args()

def main():
    args     = parse_args()
    merged   = pathlib.Path(args.merged)
    out_dir  = pathlib.Path(args.out_dir)
    llama_dir= pathlib.Path(args.llama_cpp_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not merged.exists():
        print(f"ERROR: merged model not found at {merged}")
        print("Run: python src/merge_adapter.py")
        sys.exit(1)

    # Clone llama.cpp
    if not llama_dir.exists():
        print("Cloning llama.cpp ...")
        run(["git","clone","--depth","1", LLAMA_CPP_URL, str(llama_dir)])

    # Build
    build_dir = llama_dir / "build"
    build_dir.mkdir(exist_ok=True)
    print("Building llama.cpp ...")
    run(["cmake",".."], ), cwd=str(build_dir))
    run(["cmake","--build",".","--config","Release","--parallel","4"]),  cwd=str(build_dir)

    # Convert to f16 GGUF
    fp16_gguf = out_dir / "model_f16.gguf"
    if not fp16_gguf.exists():
        script = llama_dir / "convert_hf_to_gguf.py"
        if not script.exists():
            script = llama_dir / "convert.py"
        print("Converting to GGUF ...")
        run([sys.executable, str(script), str(merged),
             "--outtype", "f16", "--outfile", str(fp16_gguf)])

    # Quantize
    quantize_bin = llama_dir / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = llama_dir / "build" / "bin" / "quantize"

    manifest = {}
    for quant_type in args.quant:
        out_gguf = out_dir / f"model_{quant_type.lower()}.gguf"
        if not out_gguf.exists():
            print(f"Quantizing to {quant_type} ...")
            run([str(quantize_bin), str(fp16_gguf), str(out_gguf), quant_type])
        size = get_size_mb(out_gguf)
        print(f"  {out_gguf.name}: {size:.1f} MB")
        print(f"  500MB gate: {'PASS' if size <= 500 else 'FAIL'}")
        print(f"  250MB gate: {'PASS' if size <= 250 else 'FAIL'}")
        manifest[quant_type] = {"path": str(out_gguf), "size_mb": round(size,1), "sha256": sha256(out_gguf)}

    manifest["primary"] = args.quant[0]
    with open(out_dir / "manifest.json","w") as f:
        json.dump(manifest, f, indent=2)
    print("Quantization complete!")


if __name__ == "__main__":
    main()