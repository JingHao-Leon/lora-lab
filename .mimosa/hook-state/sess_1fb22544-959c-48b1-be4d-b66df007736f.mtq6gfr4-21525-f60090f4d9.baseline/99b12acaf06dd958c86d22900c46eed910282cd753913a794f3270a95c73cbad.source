"""A 150-line SFT trainer: AdamW, warmup+cosine LR, grad accumulation, clipping."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .data import PackedData


@dataclass
class TrainConfig:
    lr: float = 3e-3
    epochs: int = 3
    batch_size: int = 32
    block_len: int = 64
    warmup_steps: int = 20
    grad_accum: int = 1
    clip: float = 1.0
    weight_decay: float = 0.01
    log_every: int = 50
    seed: int = 0
    device: str = "cpu"


def cosine_lr(step: int, total: int, base_lr: float, warmup: int) -> float:
    """Linear warmup then cosine decay to 10% of base_lr."""
    if step < warmup:
        return base_lr * (step + 1) / max(1, warmup)
    t = (step - warmup) / max(1, total - warmup)
    return base_lr * (0.1 + 0.45 * (1 + math.cos(math.pi * t)))


@torch.no_grad()
def evaluate(model: nn.Module, data: PackedData, cfg: TrainConfig,
             batch_size: int = 64) -> float:
    model.eval()
    losses, n = [], 0
    for i in range(0, data.x.shape[0], batch_size):
        x, y = data.x[i:i + batch_size].to(cfg.device), data.y[i:i + batch_size].to(cfg.device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                           y.reshape(-1), ignore_index=-100)
        losses.append(loss.item() * x.shape[0])
        n += x.shape[0]
    model.train()
    return sum(losses) / max(1, n)


def train(model: nn.Module, train_data: PackedData, val_data: PackedData | None,
          cfg: TrainConfig | None = None, log=print) -> dict:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    model.to(cfg.device)
    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    n_batches = math.ceil(train_data.x.shape[0] / cfg.batch_size)
    total_steps = cfg.epochs * n_batches // max(1, cfg.grad_accum)
    history = []
    t0 = time.time()
    step = 0
    for epoch in range(cfg.epochs):
        perm = torch.randperm(train_data.x.shape[0])
        for bi in range(n_batches):
            idx = perm[bi * cfg.batch_size:(bi + 1) * cfg.batch_size]
            x = train_data.x[idx].to(cfg.device)
            y = train_data.y[idx].to(cfg.device)
            logits = model(x)
            loss = nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]),
                                               y.reshape(-1), ignore_index=-100)
            (loss / cfg.grad_accum).backward()
            if (bi + 1) % cfg.grad_accum == 0:
                lr = cosine_lr(step, total_steps, cfg.lr, cfg.warmup_steps)
                for g in optim.param_groups:
                    g["lr"] = lr
                nn.utils.clip_grad_norm_(params, cfg.clip)
                optim.step()
                optim.zero_grad(set_to_none=True)
                step += 1
                if step % cfg.log_every == 0:
                    msg = f"epoch {epoch} step {step}/{total_steps} loss {loss.item():.4f} lr {lr:.2e}"
                    if val_data is not None and step % (cfg.log_every * 2) == 0:
                        msg += f" val {evaluate(model, val_data, cfg):.4f}"
                    log(msg)
                    history.append({"step": step, "loss": loss.item()})
    return {"final_loss": loss.item(),
            "val_loss": evaluate(model, val_data, cfg) if val_data is not None else None,
            "steps": step, "seconds": round(time.time() - t0, 1),
            "history": history}


@torch.no_grad()
def generate(model: nn.Module, tokenizer, prompt: str, max_new_tokens: int = 80,
             temperature: float = 0.8, device: str = "cpu") -> str:
    model.eval()
    ids = torch.tensor([tokenizer.encode(prompt, add_eos=False)], device=device)
    for _ in range(max_new_tokens):
        logits = model(ids)[:, -1, :]
        if temperature <= 0:
            nxt = logits.argmax(-1, keepdim=True)
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1)
        ids = torch.cat([ids, nxt], 1)
    return tokenizer.decode(ids[0].tolist())
