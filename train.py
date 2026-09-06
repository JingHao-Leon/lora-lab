"""End-to-end lab experiment: pretrain → LoRA SFT → compare against full SFT.

Runs on a laptop CPU in ~1-2 minutes and prints a measured comparison table:
trainable parameters, SFT val loss (baseline vs LoRA vs full), and samples.

Usage:
    uv run python train.py                # full experiment
    uv run python train.py --steps 200    # quick smoke run
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from lora_lab.data import (CharTokenizer, load_pairs, load_texts, pack_corpus,
                           pack_sft_pairs, split)
from lora_lab.lora import inject_lora, mark_only_lora_trainable, trainable_ratio
from lora_lab.model import TinyGPT
from lora_lab.trainer import TrainConfig, evaluate, generate, train

DATA = Path(__file__).parent / "data"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--sft-epochs", type=int, default=150)
    ap.add_argument("--rank", type=int, default=8)
    args = ap.parse_args()
    device = "cpu"

    # 1. tokenizer over both corpora
    general = load_texts(DATA / "general.txt")
    pairs = load_pairs(DATA / "domain_pairs.txt")
    all_text = " ".join(general) + " " + " ".join(p + " " + a for p, a in pairs)
    tok = CharTokenizer(all_text)
    print(f"vocab={tok.vocab_size} docs={len(general)} pairs={len(pairs)}")

    # 2. pretrain on general corpus
    pre = pack_corpus(tok, general, block_len=64)
    pre_train, pre_val = split(pre, 0.15)
    torch.manual_seed(0)
    model = TinyGPT(tok.vocab_size, d=128, n_heads=4, n_layers=2)
    cfg = TrainConfig(epochs=args.epochs, batch_size=16, lr=3e-3, log_every=100000)
    r = train(model, pre_train, pre_val, cfg, log=lambda m: None)
    print(f"[pretrain] train_loss={r['final_loss']:.4f} val_loss={r['val_loss']:.4f} "
          f"steps={r['steps']} time={r['seconds']}s")

    # 3. SFT data: prompt-masked pairs
    sft = pack_sft_pairs(tok, pairs, block_len=64)
    sft_train, sft_val = split(sft, 0.2)

    results = {}

    # 3a. baseline: no SFT
    results["baseline (no SFT)"] = {
        "val_loss": round(evaluate(model, sft_val, cfg), 4),
        "trainable%": 100.0,
    }

    # 3b. LoRA SFT (qkv + fc1 + fc2 heads' projections)
    torch.manual_seed(1)
    lora_model = TinyGPT(tok.vocab_size, d=128, n_heads=4, n_layers=2)
    lora_model.load_state_dict(model.state_dict())
    inject_lora(lora_model, target_modules=("qkv", "proj", "fc1", "fc2"), r=args.rank, alpha=2 * args.rank)
    mark_only_lora_trainable(lora_model)
    ratio = trainable_ratio(lora_model)
    sft_cfg = TrainConfig(epochs=args.sft_epochs, batch_size=8, lr=1e-2, log_every=100000)
    r2 = train(lora_model, sft_train, sft_val, sft_cfg, log=lambda m: None)
    results[f"LoRA SFT (r={args.rank})"] = {
        "val_loss": round(r2["val_loss"], 4), "trainable%": round(ratio * 100, 2),
    }

    # 3c. full SFT (lower LR: every weight moving on 12 samples overfits fast)
    torch.manual_seed(1)
    full_model = TinyGPT(tok.vocab_size, d=128, n_heads=4, n_layers=2)
    full_model.load_state_dict(model.state_dict())
    sft_cfg_full = TrainConfig(epochs=args.sft_epochs, batch_size=8, lr=2e-3, log_every=100000)
    r3 = train(full_model, sft_train, sft_val, sft_cfg_full, log=lambda m: None)
    results["full SFT"] = {"val_loss": round(r3["val_loss"], 4), "trainable%": 100.0}

    print(json.dumps(results, indent=2, ensure_ascii=False))

    prompt = "Q: what is lora"
    print("\n--- samples (prompt:", prompt + ") ---")
    print("baseline:", generate(model, tok, prompt, 40, temperature=0.3))
    print("lora    :", generate(lora_model, tok, prompt, 40, temperature=0.3))
    Path("runs").mkdir(exist_ok=True)
    Path("runs/results.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
