"""GPU recipe: QLoRA fine-tune a Qwen model on a real instruction dataset.

Self-contained — downloads Belle school-math instructions via HF datasets,
trains NF4-base + fp32-LoRA (the same contract implemented in lora_lab/qlora.py),
then prints before/after generations and saves loss history.

Requires a CUDA GPU. 3090 24GB measured plan: Qwen2.5-1.5B, 3000 samples,
2 epochs, bs=4 x grad_accum=4, bf16 — expect roughly 20-40 minutes.

    uv run python examples/train_qwen_gpu.py --smoke          # 100 samples, 5 min
    uv run python examples/train_qwen_gpu.py --samples 3000 --epochs 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Windows consoles often default to GBK; model output may contain any unicode
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def build_dataset(name: str, samples: int, tokenizer):
    if name == "belle":
        from datasets import load_dataset

        ds = load_dataset("BelleGroup/school_math_0.25M", split="train")
        ds = ds.shuffle(seed=0).select(range(min(samples, len(ds))))
        pairs = [(row["instruction"], row["output"]) for row in ds]
    else:  # local jsonl: {"prompt": ..., "answer": ...} per line
        pairs = []
        for line in Path(name).read_text(encoding="utf-8").splitlines():
            if line.strip():
                obj = json.loads(line)
                pairs.append((obj["prompt"], obj["answer"]))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--dataset", default="belle", help="'belle' or a local jsonl path")
    ap.add_argument("--samples", type=int, default=3000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--smoke", action="store_true", help="100 samples, 1 epoch sanity run")
    args = ap.parse_args()
    if args.smoke:
        args.samples, args.epochs = 100, 1

    import torch
    import torch.nn.functional as F
    from peft import LoraConfig, get_peft_model
    from transformers import (AutoModelForCausalLM, AutoTokenizer,
                              BitsAndBytesConfig)

    assert torch.cuda.is_available(), "this script needs a CUDA GPU"
    tok = AutoTokenizer.from_pretrained(args.model)
    bnb = BitsAndBytesConfig(  # the same NF4 contract implemented in lora_lab/qlora.py
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model, quantization_config=bnb, device_map="auto"
    )
    if "7B" in args.model or "14B" in args.model:  # trade compute for VRAM headroom
        model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(
        r=args.rank, lora_alpha=2 * args.rank, lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    pairs = build_dataset(args.dataset, args.samples, tok)
    print(f"dataset: {args.dataset} -> {len(pairs)} pairs")

    def encode(pairs):
        xs, ys = [], []
        for prompt, answer in pairs:
            p = tok(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                    add_special_tokens=False)["input_ids"]
            a = tok(answer, add_special_tokens=False)["input_ids"] + \
                [tok.eos_token_id]
            ids = (p + a)[:1024]
            x = torch.tensor(ids[:-1])
            # prompt masking, exactly as in lora_lab/data.py: only answers carry loss
            y = torch.tensor(([-100] * (len(p) - 1) + a)[: len(ids) - 1])
            xs.append(x)
            ys.append(y)
        maxlen = max(t.shape[0] for t in xs)
        pad = tok.pad_token_id or tok.eos_token_id
        x = torch.nn.utils.rnn.pad_sequence(xs, batch_first=True, padding_value=pad)
        y = torch.nn.utils.rnn.pad_sequence(ys, batch_first=True, padding_value=-100)
        return x, y

    x, y = encode(pairs)
    n_val = max(1, int(0.05 * x.shape[0]))
    xtr, ytr, xva, yva = x[n_val:], y[n_val:], x[:n_val], y[:n_val]

    from torch.utils.data import DataLoader, TensorDataset

    loader = DataLoader(TensorDataset(xtr, ytr), batch_size=args.batch_size, shuffle=True)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=2e-4, weight_decay=0.01)
    total_steps = args.epochs * len(loader) // 4  # grad_accum=4
    step = 0
    history = []

    @torch.no_grad()
    def val_loss():
        model.eval()
        outs = []
        for i in range(0, xva.shape[0], 4):
            logits = model(input_ids=xva[i:i + 4].cuda()).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   yva[i:i + 4].cuda().reshape(-1), ignore_index=-100)
            outs.append(loss.item())
        model.train()
        return sum(outs) / len(outs)

    v0 = val_loss()
    print(f"val loss before: {v0:.4f}")
    model.train()
    for epoch in range(args.epochs):
        for bi, (bx, by) in enumerate(loader):
            logits = model(input_ids=bx.cuda()).logits
            loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]).float(),
                                   by.cuda().reshape(-1), ignore_index=-100)
            (loss / 4).backward()
            if (bi + 1) % 4 == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    lr = 2e-4 * (0.1 + 0.45 * (1 + torch.cos(
                        torch.tensor(3.14159 * step / max(1, total_steps))).item()))
                    for g in opt.param_groups:
                        g["lr"] = lr
                    print(f"epoch {epoch} step {step}/{total_steps} "
                          f"loss {loss.item():.4f}")
                    history.append({"step": step, "loss": round(loss.item(), 4)})
    v1 = val_loss()
    print(f"val loss after : {v1:.4f}  (before {v0:.4f})")

    @torch.no_grad()
    def gen(prompt):
        ids = tok(f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n",
                  return_tensors="pt").input_ids.cuda()
        out = model.generate(ids, max_new_tokens=120, do_sample=False,
                             pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)

    demo = "一元二次方程 x^2 - 5x + 6 = 0 的解是什么？请给出过程。"
    try:
        print("sample after fine-tune:\n", gen(demo))
    except UnicodeEncodeError:  # never let console encoding kill the run
        print("sample skipped: console encoding cannot render it")

    model_short = args.model.split("/")[-1]  # e.g. Qwen2.5-7B-Instruct
    out_dir = Path("runs") / model_short
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir / "adapter")
    (out_dir / "results.json").write_text(json.dumps({
        "model": args.model, "dataset": args.dataset, "samples": len(pairs),
        "epochs": args.epochs, "rank": args.rank, "batch_size": args.batch_size,
        "val_loss_before": round(v0, 4), "val_loss_after": round(v1, 4),
        "history": history,
        "gpu": torch.cuda.get_device_name(0),
    }, ensure_ascii=False, indent=2))
    print(f"saved: {out_dir}/results.json + adapter/")


if __name__ == "__main__":
    main()
