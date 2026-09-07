"""Char-level SFT data pipeline: corpus → packed training tensors.

Design points that matter for SFT correctness:
- **Packing with block boundaries**: documents are joined by EOS and packed to
  fixed-length blocks; no attention isolation needed at char-lab scale.
- **Prompt masking**: for instruction pairs the prompt tokens are masked out of
  the loss (label = -100), so the model only learns to produce answers.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch


class CharTokenizer:
    """Minimal char-level tokenizer — perfect for a lab, wrong for production."""

    def __init__(self, text: str):
        self.chars = sorted(set(text)) + ["<eos>"]
        self.stoi = {c: i for i, c in enumerate(self.chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.eos_id = self.stoi["<eos>"]

    def encode(self, text: str, add_eos: bool = True) -> list[int]:
        ids = [self.stoi[c] for c in text]
        return ids + [self.eos_id] if add_eos else ids

    def decode(self, ids: list[int]) -> str:
        inv = {v: k for k, v in self.stoi.items()}
        return "".join(inv[i] for i in ids if i != self.eos_id)

    @property
    def vocab_size(self) -> int:
        return len(self.chars)


@dataclass
class PackedData:
    x: torch.Tensor  # (n_blocks, block_len)
    y: torch.Tensor  # next-token targets, -100 where masked


def pack_corpus(tokenizer: CharTokenizer, texts: list[str], block_len: int = 64) -> PackedData:
    """Concatenate with EOS separators and slice into training blocks."""
    stream: list[int] = []
    for t in texts:
        stream.extend(tokenizer.encode(t))
    n_blocks = len(stream) // (block_len + 1)
    data = torch.tensor(stream[: n_blocks * (block_len + 1)], dtype=torch.long)
    blocks = data.view(n_blocks, block_len + 1)
    return PackedData(x=blocks[:, :-1].contiguous(), y=blocks[:, 1:].contiguous())


def pack_sft_pairs(tokenizer: CharTokenizer, pairs: list[tuple[str, str]],
                   block_len: int = 64) -> PackedData:
    """Pack (prompt, answer) pairs; prompt tokens are masked with -100."""
    xs, ys = [], []
    for prompt, answer in pairs:
        p_ids = tokenizer.encode(prompt, add_eos=False)
        a_ids = tokenizer.encode(answer, add_eos=True)
        ids = p_ids + a_ids
        if len(ids) < 2:
            continue
        x = torch.tensor(ids[:-1], dtype=torch.long)
        y = torch.tensor([-100] * (len(p_ids) - 1) + a_ids, dtype=torch.long)
        # pad to block_len so batches stack
        pad = block_len - x.shape[0]
        if pad < 0:  # truncate long pairs
            x, y = x[:block_len], y[:block_len]
            pad = 0
        x = torch.nn.functional.pad(x, (0, pad), value=tokenizer.eos_id)
        y = torch.nn.functional.pad(y, (0, pad), value=-100)
        xs.append(x)
        ys.append(y)
    return PackedData(x=torch.stack(xs), y=torch.stack(ys))


def load_texts(path: str | Path) -> list[str]:
    """One document per line."""
    return [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]


def load_pairs(path: str | Path, sep: str = "=>") -> list[tuple[str, str]]:
    """One `prompt => answer` pair per line."""
    pairs = []
    for ln in Path(path).read_text(encoding="utf-8").splitlines():
        if sep in ln:
            p, a = ln.split(sep, 1)
            pairs.append((p.strip(), a.strip()))
    return pairs


def split(data: PackedData, val_frac: float = 0.1, seed: int = 0) -> tuple[PackedData, PackedData]:
    g = torch.Generator().manual_seed(seed)
    n = data.x.shape[0]
    perm = torch.randperm(n, generator=g)
    n_val = max(1, int(n * val_frac))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return (PackedData(data.x[train_idx], data.y[train_idx]),
            PackedData(data.x[val_idx], data.y[val_idx]))
