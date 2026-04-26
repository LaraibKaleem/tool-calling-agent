#!/usr/bin/env python3
import argparse, pathlib, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--adapter",    default="artifacts/lora_adapter/final_adapter")
    ap.add_argument("--output",     default="artifacts/merged_model")
    return ap.parse_args()

def main():
    args = parse_args()

    if args.base_model is None:
        meta_path = pathlib.Path(args.adapter).parent.parent / "training_meta.json"
        if meta_path.exists():
            args.base_model = json.loads(meta_path.read_text())["base_model"]
        else:
            args.base_model = "Qwen/Qwen2.5-0.5B-Instruct"

    out_dir = pathlib.Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base model : {args.base_model}")
    print(f"Adapter    : {args.adapter}")
    print(f"Output     : {out_dir}")

    print("Loading base model ...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="cpu",
        trust_remote_code=True,
    )
    
    # tokenizer = AutoTokenizer.from_pretrained(args.adapter, trust_remote_code=True)
    
    tokenizer = AutoTokenizer.from_pretrained(
    "Qwen/Qwen2.5-0.5B-Instruct",
    trust_remote_code=True
    )

    print("Loading LoRA adapter ...")
    model = PeftModel.from_pretrained(model, args.adapter)

    print("Merging weights ...")
    model = model.merge_and_unload()

    print(f"Saving merged model to {out_dir} ...")
    model.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer.save_pretrained(str(out_dir))
    print("Done!")

if __name__ == "__main__":
    main()