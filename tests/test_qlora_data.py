import pytest
import torch
import torch.nn as nn

from lora_lab.data import CharTokenizer, load_pairs, pack_corpus, pack_sft_pairs, split
from lora_lab.model import TinyGPT
from lora_lab.qlora import NF4Linear, inject_qlora, quantize_model_nf4
from lora_lab.trainer import cosine_lr


# ------------------------------------------------------------------- qlora ---

def test_nf4_storage_and_error():
    torch.manual_seed(0)
    base = nn.Linear(256, 256)
    nf4 = NF4Linear(base)
    assert nf4.idx.dtype == torch.uint8
    # ~0.5 byte/weight (4-bit idx + fp32 scale per 64-block) vs 4 bytes fp32
    assert nf4.storage_bytes() < base.weight.numel() * 4
    x = torch.randn(8, 256)
    rel = ((nf4(x) - base(x)).norm() / base(x).norm()).item()
    assert rel < 0.2  # 4-bit textbook error bound


def test_nf4_bias_preserved():
    base = nn.Linear(16, 8)
    nf4 = NF4Linear(base)
    assert nf4.bias is not None
    torch.testing.assert_close(nf4.bias, base.bias)


def test_quantize_whole_model_and_forward():
    model = TinyGPT(vocab_size=20, d=32, n_heads=2, n_layers=1)
    quantize_model_nf4(model)
    assert not any(isinstance(m, nn.Linear) for m in model.modules())
    x = torch.randint(0, 20, (2, 16))
    out = model(x)
    assert out.shape == (2, 16, 20)
    assert torch.isfinite(out).all()


def test_qlora_adapters_train():
    torch.manual_seed(0)
    model = TinyGPT(vocab_size=20, d=32, n_heads=2, n_layers=1)
    quantize_model_nf4(model)
    inject_qlora(model, target_modules=("qkv",), r=2)
    x = torch.randint(0, 20, (2, 16))
    loss = model(x).sum()
    loss.backward()
    lora_grads = [n for n, p in model.named_parameters()
                  if "lora_" in n and p.grad is not None and p.grad.abs().sum() > 0]
    assert lora_grads


# -------------------------------------------------------------------- data ---

def test_tokenizer_roundtrip():
    tok = CharTokenizer("hello 世界")
    text = "hello 世界"
    assert tok.decode(tok.encode(text)) == text
    assert tok.eos_id == tok.vocab_size - 1


def test_pack_corpus_token_conservation():
    tok = CharTokenizer("abcdef")
    texts = ["abc", "def", "abcde"]
    packed = pack_corpus(tok, texts, block_len=8)
    n_tokens = sum(len(tok.encode(t)) for t in texts)
    assert packed.x.shape[0] == n_tokens // 9  # (block_len+1) per block
    # every target is the next input token
    torch.testing.assert_close(packed.y[:, 0], packed.x[:, 1])


def test_pack_sft_pairs_masks_prompt():
    tok = CharTokenizer("abcdef=> ")
    pairs = [("abc", "def"), ("ab", "cdef")]
    packed = pack_sft_pairs(tok, pairs, block_len=8)
    assert packed.x.shape == packed.y.shape
    masked = (packed.y == -100)
    assert masked.any()  # prompts are masked
    answers = packed.y[packed.y != -100]
    assert (answers != -100).all()
    # block 0: prompt 'abc' → first 2 targets masked, answer tokens unmasked,
    # trailing pad (-100) does not count as prompt
    assert (packed.y[0][:2] == -100).all()
    assert (packed.y[0][2:6] != -100).all()


def test_split_deterministic():
    tok = CharTokenizer("abcdef")
    packed = pack_corpus(tok, ["abcdef" * 10], block_len=8)
    tr1, va1 = split(packed, seed=0)
    tr2, va2 = split(packed, seed=0)
    assert torch.equal(tr1.x, tr2.x) and torch.equal(va1.x, va2.x)
    assert tr1.x.shape[0] + va1.x.shape[0] == packed.x.shape[0]


# ----------------------------------------------------------------- trainer ---

def test_cosine_schedule_shape():
    assert cosine_lr(0, 100, 1.0, 10) == pytest.approx(0.1)  # warmup start
    assert cosine_lr(9, 100, 1.0, 10) == pytest.approx(1.0)  # warmup peak
    mid = cosine_lr(55, 100, 1.0, 10)
    end = cosine_lr(99, 100, 1.0, 10)
    assert mid > end >= 0.099  # decays toward 0.1*base


def test_grad_clip_prevents_explosion():
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.zeros(8))]
    torch.nn.utils.clip_grad_norm_(params, 1.0)  # no-op on zero grads
    params[0].grad = torch.full((8,), 100.0)
    torch.nn.utils.clip_grad_norm_(params, 1.0)
    assert params[0].grad.norm().item() <= 1.0 + 1e-6
