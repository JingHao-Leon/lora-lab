"""GPU recipe: QLoRA fine-tune a Qwen model on an instruction dataset.

This script is the *production-scale* counterpart of the lab: it uses the same
concepts (NF4 base + fp32 LoRA, prompt masking, cosine schedule) through
HuggingFace ecosystems. It requires a CUDA GPU (>=16 GB) and is NOT run by the
test suite — the lab itself (../train.py) proves the numerics on CPU.

Usage:
    pip install transformers peft datasets bitsandbytes accelerate
    python examples/train_qwen_gpu.py --model Qwen/Qwen2.5-1.5B --data data/domain_pairs.jsonl
"""

import argparse


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    ap.add_argument("--data", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=3)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig, Trainer, TrainingArguments)

    bnb = BitsAndBytesConfig(  # the same NF4 contract implemented in lora_lab/qlora.py
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=bnb,
                                                 device_map="auto")
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    ds = load_dataset("json", data_files=args.data, split="train")

    def collate(batch):
        xs, ys = [], []
        for row in batch:
            p = tok(f"Q: {row['prompt']}\nA: ", add_special_tokens=False)["input_ids"]
            a = tok(row["answer"], add_special_tokens=False)["input_ids"] + [tok.eos_token_id]
            ids = p + a
            xs.append(ids[:-1])
            ys.append([-100] * (len(p) - 1) + a)  # prompt masking, as in data.py
        maxlen = max(map(len, xs))
        pad = tok.pad_token_id or tok.eos_token_id
        import torch.nn.functional as F
        x = torch.stack([F.pad(t, (0, maxlen - len(t)), value=pad) for t in xs])
        y = torch.stack([F.pad(t, (0, maxlen - len(t)), value=-100) for t in ys])
        return {"input_ids": x, "labels": y}

    Trainer(
        model=model,
        train_dataset=ds,
        data_collator=collate,
        args=TrainingArguments(
            output_dir="runs/qwen-qlora",
            num_train_epochs=args.epochs,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=4,
            learning_rate=2e-4,
            lr_scheduler_type="cosine",
            warmup_steps=50,
            bf16=True,
            logging_steps=10,
        ),
    ).train()
    model.save_pretrained("runs/qwen-qlora/adapter")


if __name__ == "__main__":
    main()
