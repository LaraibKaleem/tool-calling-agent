"""
inference.py
============
Exposes:  def run(prompt: str, history: list[dict]) -> str
No network imports allowed.
"""

import os
import json
import re
import pathlib
from typing import Optional

ARTIFACTS_DIR   = pathlib.Path(__file__).parent / "artifacts"
QUANT_DIR       = ARTIFACTS_DIR / "quantized"
ADAPTER_DIR     = ARTIFACTS_DIR / "lora_adapter" / "final_adapter"
MERGED_DIR      = ARTIFACTS_DIR / "merged_model"

GGUF_MODEL_PATH = os.environ.get("GGUF_MODEL_PATH", "")
HF_MODEL_PATH   = os.environ.get("HF_MODEL_PATH", "")
BASE_MODEL_ID   = os.environ.get("BASE_MODEL_ID", "Qwen/Qwen2.5-0.5B-Instruct")

MAX_NEW_TOKENS  = 128
TEMPERATURE     = 0.01
TOP_P           = 0.9
N_CTX           = 1024

SYSTEM_PROMPT = """You are a mobile assistant. When the user's request maps to one of the available tools, respond ONLY with a tool call in this exact format (no other text):
<tool_call>{"tool": "TOOL_NAME", "args": {ARGS_JSON}}</tool_call>

Available tools:
- weather:  {"tool": "weather",  "args": {"location": "string", "unit": "C|F"}}
- calendar: {"tool": "calendar", "args": {"action": "list|create", "date": "YYYY-MM-DD", "title": "string (required for create)"}}
- convert:  {"tool": "convert",  "args": {"value": number, "from_unit": "string", "to_unit": "string"}}
- currency: {"tool": "currency", "args": {"amount": number, "from": "ISO3", "to": "ISO3"}}
- sql:      {"tool": "sql",      "args": {"query": "string"}}

If no tool fits, respond naturally in plain text. Do NOT use <tool_call> tags for refusals."""

# ── Find model files ──────────────────────────────────────────────────────────
def _find_gguf() -> Optional[str]:
    if GGUF_MODEL_PATH and pathlib.Path(GGUF_MODEL_PATH).exists():
        return GGUF_MODEL_PATH
    manifest_path = QUANT_DIR / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        primary  = manifest.get("primary", "Q4_K_M")
        entry    = manifest.get(primary, {})
        p        = entry.get("path", "")
        if p and pathlib.Path(p).exists():
            return p
    for candidate in sorted(QUANT_DIR.glob("*.gguf")):
        if "f16" not in candidate.name:
            return str(candidate)
    return None

def _find_hf_model() -> Optional[str]:
    if HF_MODEL_PATH and pathlib.Path(HF_MODEL_PATH).exists():
        return HF_MODEL_PATH
    for p in [MERGED_DIR, ADAPTER_DIR]:
        if p.exists() and any(p.glob("*.safetensors")):
            return str(p)
    return None

# ── Backend 1: llama-cpp-python ───────────────────────────────────────────────
_llama_model = None

def _load_llama_cpp(gguf_path: str):
    global _llama_model
    if _llama_model is not None:
        return _llama_model
    from llama_cpp import Llama
    _llama_model = Llama(
        model_path=gguf_path,
        n_ctx=N_CTX,
        n_threads=os.cpu_count() or 4,
        n_gpu_layers=0,
        verbose=False,
    )
    return _llama_model

def _run_llama_cpp(prompt_text: str, gguf_path: str) -> str:
    model  = _load_llama_cpp(gguf_path)
    output = model(
        prompt_text,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        stop=["<|im_end|>", "<|endoftext|>", "</s>"],
        echo=False,
    )
    return output["choices"][0]["text"].strip()

# ── Backend 2: HuggingFace transformers ──────────────────────────────────────
_hf_model     = None
_hf_tokenizer = None

def _load_hf(model_path: str):
    global _hf_model, _hf_tokenizer
    if _hf_model is not None:
        return _hf_model, _hf_tokenizer
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    _hf_tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )
    if _hf_tokenizer.pad_token is None:
        _hf_tokenizer.pad_token = _hf_tokenizer.eos_token
    try:
        from transformers import BitsAndBytesConfig
        bnb = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16
        )
        _hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=bnb,
            device_map="auto",
            trust_remote_code=True,
        )
    except Exception:
        _hf_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="cpu",
            torch_dtype=torch.float32,
            trust_remote_code=True,
        )
    _hf_model.eval()
    return _hf_model, _hf_tokenizer

def _run_hf(messages: list, model_path: str) -> str:
    import torch
    model, tokenizer = _load_hf(model_path)
    text   = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            top_p=TOP_P,
            do_sample=TEMPERATURE > 0,
            pad_token_id=tokenizer.eos_token_id,
        )
    generated = out[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(generated, skip_special_tokens=True).strip()

# ── Prompt builder ────────────────────────────────────────────────────────────
def _build_messages(prompt: str, history: list) -> list:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": prompt})
    return messages

def _build_chatml_string(messages: list) -> str:
    out = ""
    for m in messages:
        out += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
    out += "<|im_start|>assistant\n"
    return out

# ── Clean output ──────────────────────────────────────────────────────────────
_TAG_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)

def _clean_output(raw: str) -> str:
    raw = raw.strip()
    m   = _TAG_RE.search(raw)
    if not m:
        return raw
    try:
        json.loads(m.group(1).strip())
        return m.group(0)
    except json.JSONDecodeError:
        return raw

# ── Main function ─────────────────────────────────────────────────────────────
def run(prompt: str, history: list) -> str:
    messages = _build_messages(prompt, history)

    # Try GGUF first
    gguf_path = _find_gguf()
    if gguf_path:
        try:
            chatml = _build_chatml_string(messages)
            raw    = _run_llama_cpp(chatml, gguf_path)
            return _clean_output(raw)
        except Exception as e:
            print(f"[inference] llama-cpp failed: {e}")

    # Try HF model
    hf_path = _find_hf_model()
    if hf_path:
        try:
            raw = _run_hf(messages, hf_path)
            return _clean_output(raw)
        except Exception as e:
            print(f"[inference] HF failed: {e}")

    return "I'm unable to process that request right now."


# ── Smoke test ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import time
    tests = [
        ("What's the weather in Paris?", []),
        ("Convert 100 USD to EUR",       []),
        ("Tell me a joke",               []),
    ]
    for prompt, history in tests:
        t0  = time.perf_counter()
        out = run(prompt, history)
        ms  = (time.perf_counter() - t0) * 1000
        print(f"[{ms:6.1f}ms] {prompt}")
        print(f"         {out}\n")