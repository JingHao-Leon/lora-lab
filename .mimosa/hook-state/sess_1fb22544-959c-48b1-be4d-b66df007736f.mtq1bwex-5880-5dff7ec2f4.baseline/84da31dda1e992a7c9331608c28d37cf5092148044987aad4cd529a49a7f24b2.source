"""Tiny char-level GPT for the lab: small enough to train on a laptop CPU."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Head(nn.Module):
    def __init__(self, d: int, hd: int):
        super().__init__()
        self.qkv = nn.Linear(d, 3 * hd, bias=False)
        self.proj = nn.Linear(hd, d, bias=False)
        self.hd = hd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).split(self.hd, dim=-1)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.hd)
        mask = torch.ones(T, T, dtype=torch.bool, device=x.device).tril()
        att = F.softmax(att.masked_fill(~mask, float("-inf")), dim=-1)
        return self.proj(att @ v)


class Block(nn.Module):
    def __init__(self, d: int, n_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.heads = nn.ModuleList(Head(d, d // n_heads) for _ in range(n_heads))
        self.ln2 = nn.LayerNorm(d)
        self.fc1 = nn.Linear(d, mlp_ratio * d)
        self.fc2 = nn.Linear(mlp_ratio * d, d)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.ln1(x)
        x = x + sum(head(h) for head in self.heads)
        x = x + self.fc2(F.gelu(self.fc1(self.ln2(x))))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d: int = 128, n_heads: int = 4,
                 n_layers: int = 3, block_len: int = 64):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d)
        self.pos = nn.Embedding(block_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads) for _ in range(n_layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab_size, bias=False)
        self.block_len = block_len

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        assert T <= self.block_len
        pos = torch.arange(T, device=idx.device)
        x = self.tok(idx) + self.pos(pos)[None]
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))
