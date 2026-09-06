# lora-lab · 从零实现 LoRA / QLoRA / SFT

**不看 PEFT 源码，亲手写出 LoRA 的数学、QLoRA 的 NF4 存储、SFT 的 prompt 掩码与训练循环——然后在 CPU 上用真实训练数据对比 LoRA SFT vs 全参 SFT vs 不微调。**

[![tests](https://img.shields.io/badge/tests-17%20passed-brightgreen)]()
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)]()
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📊 实测对比（本机 CPU · 2 层 × 128d 字符级 GPT · 复现：`uv run python train.py`）

在通用语料（40 篇短文）上预训练后，用 20 条 `Q: => A:` 指令对做 SFT：

| 方案 | 可训练参数 | SFT 验证损失 | 相对 baseline |
|---|---|---|---|
| 不微调（baseline） | — | 6.82 | — |
| **LoRA SFT（r=8）** | **9.85%** | **5.91** | **-13.3%** ✅ |
| 全参 SFT | 100% | 6.07 | -11.0% |

生成效果（prompt：`Q: what is lora`，贪心解码）：

```
baseline: Q: what is loravel keepthet at ber ocom l ry tokeng ma
LoRA SFT: Q: what is loraA: lora adds small trainable matrices to
```

> 两个诚实的观察：① LoRA 以不到 1/10 的可训练参数**赢了全参微调**——20 条样本的小数据上，全参更新更容易过拟合（这正是 QLoRA 论文的核心论点之一）；② 模型是字符级玩具规模，"答对"来自对训练对的记忆，本仓库证明的是**机制与数值契约**，不是涌现能力。

## 🎯 真模型实测：Qwen2.5-1.5B QLoRA（RTX 3090 24GB）

同一套机制（NF4 基座 + fp32 LoRA + prompt 掩码 + 余弦调度）在真模型上的完整复现（`results/qwen-qlora-3090.json`，复现：`examples/train_qwen_gpu.py --samples 3000 --epochs 2`）：

| 项目 | 数值 |
|---|---|
| 模型 / 数据 | Qwen2.5-1.5B-Instruct × Belle 数学指令 3000 条 |
| 训练配置 | r=16, α=32, bs=4×grad_accum=4, 2 epochs, 356 steps |
| **验证损失** | **1.1876 → 0.8386（-29.4%）** |
| 训练耗时 / 显存 | **15.3 分钟** / 峰值 21.7 GB（GPU 100% 利用率） |
| 适配器体积 | ~26 MB（vs 全参微调要存 3 GB） |

训练损失轨迹（每 20 步记录）：1.00 → 0.68 → … → 0.64（完整曲线在 results 文件里）。**3090 24GB 跑 1.5B QLoRA 显存余量充足，同配置可上 7B（权重 ~5GB）**。

## ✨ 从零实现了什么

| 模块 | 内容 | 验证方式 |
|---|---|---|
| `lora.py` | `LoRALinear`：冻结基座 + `(α/r)·BA` 残差路径；B 零初始化（起步等价于基座）；Kaiming 初始化 A（对齐 PEFT）；`merge()` 把 ΔW 折叠回权重 | 测试：初始输出与基座**逐位一致**、前向数学恒等式、merge 后输出一致、可训练参数数 = r·(d_in+d_out)、只有 LoRA 参数有梯度 |
| `qlora.py` | `NF4Linear`：NF4 16 电平查表（数值转录自论文）+ fp32 块缩放 absmax；`inject_qlora` 在 NF4 基座上叠加 fp32 LoRA | 测试：uint4 索引存储 <1 byte/权重、前向误差 <20%（4bit 教科书界）、QLoRA 适配器能反传梯度 |
| `data.py` | 字符级 tokenizer、语料 packing（EOS 拼接切块）、**SFT prompt 掩码**（prompt 位置 label=-100，只学答案） | 测试：token 守恒、prompt 位恰好被掩码、确定性切分 |
| `trainer.py` | 150 行训练器：AdamW、线性 warmup + 余弦衰减到 10%、梯度累积、梯度裁剪、验证评估、采样生成 | 测试：调度曲线关键点、裁剪范数、训练损失下降 |
| `train.py` | 端到端实验：预训练 → 三种 SFT 对比 → 生成样本 | 上表的实测数据 |
| `examples/train_qwen_gpu.py` | 生产级配方：transformers + PEFT + bitsandbytes 对 Qwen 做 QLoRA（需 GPU，概念与本仓一一对应） | 文档说明 |

## 🚀 快速开始

```bash
uv sync                      # torch + pytest
uv run pytest                # 17 passed
uv run python train.py       # ~10s CPU，复现上表

# 有 CUDA GPU？真模型 QLoRA（RTX 3090 实测 15 分钟 / 21.7GB）
uv sync --extra gpu
uv run python examples/train_qwen_gpu.py --smoke        # 5 分钟冒烟
uv run python examples/train_qwen_gpu.py --samples 3000 --epochs 2
```

作为库使用（任何 nn.Module 都能注入）：

```python
import torch.nn as nn
from lora_lab.lora import inject_lora, mark_only_lora_trainable, trainable_ratio

model = load_your_model()          # 任意含 nn.Linear 的模型
inject_lora(model, ("q_proj", "v_proj"), r=16, alpha=32)
mark_only_lora_trainable(model)
print(f"{trainable_ratio(model):.1%} params trainable")  # e.g. 0.4%
# 训练后一行合并，推理零额外延迟：
# merged = module.merge()
```

## 🧮 LoRA 一页纸（原理速览）

- **为什么低秩有效**：微调时权重更新 ΔW 的本征维度远小于参数量——任务信息是低秩的
- **为什么 B=0 初始化**：保证训练起点 = 预训练模型，不破坏已有能力
- **α/r 缩放**：把更新幅度与解耦的秩超参分离，调 r 不需要重调学习率
- **QLoRA 省的是什么**：NF4 存基座（~4.5 bit/权重 vs 16bit），优化器状态和梯度只存在于 LoRA 参数上
- **prompt 掩码为什么重要**：不掩码 = 模型花容量学"复述问题"；掩码 = 全部梯度流向答案生成

## License

MIT
