#!/usr/bin/env python3
import os, json, argparse, pathlib
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from trl import   SFTTrainer, SFTConfig

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEFAULT_OUTPUT_DIR = "artifacts/lora_adapter"
DEFAULT_DATA_DIR   = "data"

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model",  default=DEFAULT_BASE_MODEL)
    ap.add_argument("--output-dir",  default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--data-dir",    default=DEFAULT_DATA_DIR)
    ap.add_argument("--epochs",      type=int,   default=3)
    ap.add_argument("--batch-size",  type=int,   default=4)
    ap.add_argument("--grad-accum",  type=int,   default=4)
    ap.add_argument("--lr",          type=float, default=2e-4)
    ap.add_argument("--lora-rank",   type=int,   default=16)
    ap.add_argument("--lora-alpha",  type=int,   default=32)
    ap.add_argument("--max-seq-len", type=int,   default=512)
    ap.add_argument("--seed",        type=int,   default=42)
    return ap.parse_args()

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f if l.strip()]

def main():
    args    = parse_args()
    out_dir = pathlib.Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    use_4bit   = torch.cuda.is_available()
    device_map = "auto" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device_map} | 4bit: {use_4bit} | Model: {args.base_model}")

    print("Loading tokenizer ...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model, trust_remote_code=True, padding_side="right"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("Loading model ...")
    bnb_config = None
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16 if not use_4bit else None,
        attn_implementation="eager",
    )
    model.config.use_cache = False
    if use_4bit:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Loading data ...")
    train_raw = load_jsonl(str(pathlib.Path(args.data_dir) / "train.jsonl"))
    val_raw   = load_jsonl(str(pathlib.Path(args.data_dir) / "val.jsonl"))
    train_ds  = Dataset.from_list(train_raw)
    val_ds    = Dataset.from_list(val_raw)

    def format_messages(example):
        text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return {"text": text}

    train_ds = train_ds.map(format_messages, remove_columns=["messages","id"])
    val_ds   = val_ds.map(format_messages,   remove_columns=["messages","id"])
    print(f"Train: {len(train_ds)}  Val: {len(val_ds)}")

    fp16 = torch.cuda.is_available() and not torch.cuda.is_bf16_supported()
    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

    training_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.05,
        fp16=fp16,
        bf16=bf16,
        logging_steps=10,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        dataloader_num_workers=0,
        optim="paged_adamw_8bit" if use_4bit else "adamw_torch",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_grad_norm=1.0,
        weight_decay=0.01,
    )

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     args=training_args,
    #     train_dataset=train_ds,
    #     eval_dataset=val_ds,
    #     dataset_text_field="text",
    #     max_seq_length=args.max_seq_len,
    #     packing=True,
    #     callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    # )
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        packing=True,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )
    # sft_config = SFTConfig(
    # output_dir="artifacts/lora_adapter",
    # num_train_epochs=3,
    # per_device_train_batch_size=4,
    # gradient_accumulation_steps=4,
    # learning_rate=2e-4,
    # logging_steps=10,
    # save_steps=200,
    # max_seq_length=512,   # ✅ NOW goes here
    # packing=True
    # )

    # trainer = SFTTrainer(
    #     model=model,
    #     tokenizer=tokenizer,
    #     train_dataset=train_dataset,
    #     eval_dataset=val_dataset,
    #     args=sft_config,
    # )

    print("\nStarting training ...")
    trainer.train()

    adapter_path = out_dir / "final_adapter"
    adapter_path.mkdir(exist_ok=True)
    print(f"Saving adapter to {adapter_path} ...")
    model.save_pretrained(str(adapter_path))
    tokenizer.save_pretrained(str(adapter_path))

    with open(out_dir / "training_meta.json","w") as f:
        json.dump({"base_model": args.base_model, "lora_rank": args.lora_rank}, f)

    print("Training complete!")

if __name__ == "__main__":
    main()