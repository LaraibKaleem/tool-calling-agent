<!-- # 🔧 Tool-Calling Mobile Assistant

A fine-tuned **Qwen2.5-0.5B-Instruct** model that performs structured 
tool calls for an on-device mobile assistant. Runs fully offline, 
fits under 500 MB after quantization, and achieves under 200ms per 
turn on CPU.

---

## What This Project Does

When you say something like:
> "What's the weather in Paris?"

The model outputs:<tool_call>{"tool": "weather", "args": {"location": "Paris", "unit": "C"}}</tool_call>
Your app reads that and fetches the weather. No internet needed.

---

## 5 Tools Supported

| Tool | Example |
|---|---|
| Weather | "How hot is Dubai?" |
| Calendar | "Schedule a meeting on June 3rd" |
| Unit Convert | "5 kg to pounds" |
| Currency | "500 USD to EUR" |
| SQL | "Show users older than 25" |

---

## Setup — Step by Step

### Step 1 — Clone the repo
git clone https://github.com/LaraibKaleem/tool-calling-agent.git
cd tool-calling-agent

### Step 2 — Install packages
pip install -r requirements.txt

### Step 3 — Generate training data
python src/generate_data.py --n 2000

### Step 4 — Train the model (needs GPU)
python src/train.py

### Step 5 — Merge adapter
python src/merge_adapter.py

### Step 6 — Quantize
python src/quantize.py --quant Q4_K_M

### Step 7 — Evaluate
python src/evaluate.py --latency --verbose

### Step 8 — Run demo
python demo/app.py
Open browser at http://localhost:7860

### Run everything at once
make all
---

## Run on Google Colab (Free GPU)

1. Go to colab.research.google.com
2. Open notebooks/colab_full_pipeline.ipynb
3. Runtime → Change runtime type → T4 GPU
4. Run all cells top to bottom
5. Total time: about 30 minutes

---

## Project Structure
tool-call-finetune/
│
├── inference.py              ← grader calls this file
├── README.md                 ← this file
├── Makefile                  ← run make all
├── requirements.txt          ← packages for running
├── requirements-train.txt    ← packages for training
├── .gitignore                ← files to ignore
│
├── src/
│   ├── generate_data.py      ← creates 2000 training examples
│   ├── train.py              ← fine-tunes model with LoRA
│   ├── merge_adapter.py      ← merges LoRA into base model
│   ├── quantize.py           ← shrinks model to GGUF format
│   └── evaluate.py           ← tests accuracy on examples
│
├── demo/
│   └── app.py                ← Gradio chat demo in browser
│
├── starter/
│   ├── public_test.jsonl     ← 40 test examples
│   ├── teacher_examples.jsonl← 20 seed examples
│   ├── tool_schemas.json     ← 5 tool definitions
│   └── eval_harness_contract.py ← grader scoring code
│
└── notebooks/
└── colab_full_pipeline.ipynb ← full Colab notebook

---

## Evaluation Results

| Slice           | Examples | Score |
|-----------------|----------|-------|
| A Standard      | 16       | 0.94  |
| B Paraphrased   | 10       | 0.90  |
| C Adversarial   | 10       | 0.82  |
| D Refusals      | 4        | 0.88  |
| Overall         | 40       | 0.89  |

---

## Training Data

Total examples: 2000 generated automatically using templates.

| Type                  | Percentage |
|-----------------------|------------|
| Weather calls         | 20%        |
| Unit conversions      | 15%        |
| Currency conversions  | 13%        |
| Calendar list         | 9%         |
| Calendar create       | 9%         |
| SQL queries           | 8%         |
| Refusals              | 12%        |
| Adversarial examples  | 9%         |
| Multi-turn            | 5%         |

---

## Why Qwen2.5-0.5B?

1. Passes all size gates at Q4_K_M (310 MB)
2. Strong JSON output from code pretraining
3. Fast CPU inference at 140 tokens per second
4. Good multilingual base for adversarial prompts
5. Free and open source

Other models tested and why rejected:

| Model              | Problem                        |
|--------------------|--------------------------------|
| Qwen2.5-1.5B       | 900 MB — fails 500 MB gate     |
| TinyLlama-1.1B     | 700 MB — fails 500 MB gate     |
| SmolLM2-360M       | Too small, poor JSON output    |
| Gemma-2-2B         | 1.3 GB — fails 500 MB gate     |

---

## Why LoRA rank 16?

- Rank 8 failed on ISO currency codes (EUR became ERU)
- Rank 16 targeting all projections fixed this
- Good balance of adapter size vs accuracy

---

## What Worked

- Qwen2.5-0.5B has great JSON accuracy after fine-tuning
- Sequence packing gave 2x training speed
- Template data generation needs no API key
- llama-cpp-python gives 140 tok/sec on CPU
- System prompt with explicit refusal triggers

---

## What Did Not Work

- LoRA rank 8 — failed on currency ISO codes
- Less than 500 training examples — model memorized
- SmolLM2 as base — worse JSON than Qwen
- Pure greedy decoding — repetitive SQL outputs
- QLoRA merged directly to GGUF — produced garbage

---


 -->
