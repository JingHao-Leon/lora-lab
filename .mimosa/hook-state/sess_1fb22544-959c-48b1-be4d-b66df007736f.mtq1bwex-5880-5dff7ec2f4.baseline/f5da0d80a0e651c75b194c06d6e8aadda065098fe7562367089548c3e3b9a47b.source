import torch
import torch.nn as nn

from lora_lab.data import CharTokenizer, load_pairs, pack_corpus, pack_sft_pairs
from lora_lab.lora import (LoRALinear, inject_lora, mark_only_lora_trainable,
                           trainable_ratio)
from lora_lab.model import TinyGPT
from lora_lab.trainer import cosine_lr, evaluate, train, TrainConfig

import pytest


# ------------------------------------------------------------------- lora ---

def test_lora_init_is_identity():
    """B=0 at init → LoRA output must equal base output exactly."""
    torch.manual_seed(0)
    base = nn.Linear(32, 16)
    lora = LoRALinear(base, r=4)
    x = torch.randn(3, 32)
    torch.testing.assert_close(lora(x), base(x))


def test_lora_forward_math():
    lora = LoRALinear(nn.Linear(8, 8), r=2, alpha=4)
    with torch.no_grad():
        lora.lora_B.normal_()
    x = torch.randn(4, 8)
    expected = lora.base(x) + (x @ lora.lora_A.T @ lora.lora_B.T) * (4 / 2)
    torch.testing.assert_close(lora(x), expected)


def test_base_weight_frozen():
    lora = LoRALinear(nn.Linear(8, 8), r=2)
    assert not lora.base.weight.requires_grad
    assert lora.lora_A.requires_grad and lora.lora_B.requires_grad


def test_merge_equals_adapter_output():
    torch.manual_seed(0)
    lora = LoRALinear(nn.Linear(24, 16), r=4, alpha=8)
    with torch.no_grad():
        lora.lora_B.normal_(std=0.1)
    x = torch.randn(5, 24)
    merged = lora.merge()
    torch.testing.assert_close(merged(x), lora(x), rtol=1e-5, atol=1e-5)


def test_trainable_param_count():
    """r=4 on a 64×32 linear: 4*(64+32) = 384 trainable vs 2048+64 base."""
    lora = LoRALinear(nn.Linear(32, 64), r=4)
    n = sum(p.numel() for p in lora.parameters() if p.requires_grad)
    assert n == 4 * (32 + 64)
    assert lora.delta_weight().shape == (64, 32)


def test_inject_and_train_only_lora():
    model = TinyGPT(vocab_size=30, d=32, n_heads=2, n_layers=1)
    inject_lora(model, target_modules=("qkv",), r=2)
    mark_only_lora_trainable(model)
    ratio = trainable_ratio(model)
    assert 0 < ratio < 0.5
    grads = []
    x = torch.randint(0, 30, (2, 16))
    model(x).sum().backward()
    for name, p in model.named_parameters():
        if "lora_" in name:
            assert p.grad is not None, name
            grads.append(name)
        else:
            assert p.grad is None or p.grad.abs().sum() == 0, name
    assert grads  # at least the qkv adapters trained


def test_lora_training_reduces_loss():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=20, d=32, n_heads=2, n_layers=1)
    inject_lora(model, target_modules=("qkv", "fc1"), r=4)
    mark_only_lora_trainable(model)
    x = torch.randint(0, 20, (32, 32))
    data = type("D", (), {"x": x, "y": torch.roll(x, -1, dims=1)})()
    before = evaluate(model, data, TrainConfig())
    train(model, data, None, TrainConfig(epochs=3, batch_size=16, lr=5e-3, log_every=999),
          log=lambda m: None)
    after = evaluate(model, data, TrainConfig())
    assert after < before
