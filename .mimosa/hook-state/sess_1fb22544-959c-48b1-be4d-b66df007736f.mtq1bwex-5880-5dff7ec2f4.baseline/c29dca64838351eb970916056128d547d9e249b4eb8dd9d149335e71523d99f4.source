"""LoRA (Low-Rank Adaptation) from scratch — no PEFT dependency.

y = W₀x + (α/r)·B(Ax), with W₀ frozen and A ∈ R^{r×d}, B ∈ R^{d_out×r}.
Only A and B receive gradients: trainable params drop from d·d_out to r·(d+d_out).
``merge`` folds ΔW = (α/r)·BA back into the weight for zero-latency inference.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wraps an nn.Linear with a trainable low-rank residual path."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: float = 16.0,
                 dropout: float = 0.0):
        super().__init__()
        assert r > 0
        self.base = base
        self.r, self.alpha = r, alpha
        self.scaling = alpha / r
        for p in self.base.parameters():
            p.requires_grad_(False)  # freeze the pretrained weight
        self.lora_A = nn.Parameter(torch.zeros(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))  # PEFT-compatible init
        # B stays zero → ΔW = 0 at start, model output is exactly the base's
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.dropout(x) @ self.lora_A.T @ self.lora_B.T * self.scaling

    def delta_weight(self) -> torch.Tensor:
        return self.lora_B @ self.lora_A * self.scaling

    def merge(self) -> nn.Linear:
        """Fold LoRA into a plain Linear (weights += ΔW)."""
        merged = nn.Linear(self.base.in_features, self.base.out_features,
                           bias=self.base.bias is not None)
        with torch.no_grad():
            merged.weight.copy_(self.base.weight + self.delta_weight())
            if self.base.bias is not None:
                merged.bias.copy_(self.base.bias)
        return merged

    def trainable_parameters(self) -> list[nn.Parameter]:
        return [self.lora_A, self.lora_B]


def inject_lora(model: nn.Module, target_modules: tuple[str, ...] = ("qkv", "proj"),
                r: int = 8, alpha: float = 16.0, dropout: float = 0.0) -> nn.Module:
    """Replace every nn.Linear whose name contains a target substring with LoRALinear."""
    replaced = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if any(t in child_name for t in target_modules) and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
                replaced.append(f"{name}.{child_name}" if name else child_name)
    assert replaced, f"no module matched targets {target_modules}"
    model._lora_targets = replaced  # type: ignore[attr-defined]
    return model


def trainable_ratio(model: nn.Module) -> float:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable / total


def mark_only_lora_trainable(model: nn.Module) -> nn.Module:
    """Freeze everything, then unfreeze LoRA params (call after inject_lora)."""
    for p in model.parameters():
        p.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.lora_A.requires_grad_(True)
            module.lora_B.requires_grad_(True)
    return model
