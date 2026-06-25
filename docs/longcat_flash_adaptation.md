# LongCat-Flash-Chat 模型适配文档

## 概述

本文档记录了在 torchtitan-npu 框架中适配 `meituan-longcat/LongCat-Flash-Chat` 模型的完整过程。LongCat-Flash-Chat 是美团发布的稀疏 MoE+MLA 架构大模型，采用 Multi-head Latent Attention（源自 DeepSeek MLA）和 512 路由专家 + 256 零专家的混合专家设计。

**模型关键参数（完整版）：**

| 参数 | 值 |
|------|-----|
| hidden_size | 6144 |
| num_layers (macro-blocks) | 28 |
| num_attention_heads | 64 |
| vocab_size | 131072 |
| num_routed_experts | 512 |
| num_zero_experts | 256 |
| moe_top_k | 12 |
| attention_method | MLA (Multi-head Latent Attention) |
| q_lora_rank | 1536 |
| kv_lora_rank | 512 |
| max_position_embeddings | 131072 |

## 实现的代码

适配代码位于 `torchtitan_npu/models/longcat_flash/`，包含以下文件：

### 1. `model.py` — 模型定义

核心模块实现：

- **`LongCatFlashModel`** — 顶层模型，包含 embedding、macro-block 层、final norm 和 output projection
- **`LongCatFlashMacroBlock`** — 宏块，每个包含 2 个 MLA attention + 2 个 FFN + 1 个 MoE（shortcut 结构）
- **`LongCatFlashMLA`** — Multi-head Latent Attention，含 q/kv 的 LoRA 压缩、RoPE 和 SDPA，支持 absorb 模式
- **`LongCatFlashMoE`** — 混合专家层，实现 softmax routing + e_score_correction_bias + identity zero-expert + NPU token dispatch
- **`LongCatFlashExperts`** — 专家计算，使用 **w13 融合**（gate_proj + up_proj 合并为单参数）+ `grouped_mm` + `npu_swiglu`

### 2. `parallelize.py` — 并行化策略

- **Tensor Parallel (TP)**：MLA 的 q_b_proj/kv_b_proj ColwiseParallel, o_proj RowwiseParallel, FFN ColwiseParallel/RowwiseParallel
- **Expert Parallel (EP)**：通过 `ExpertParallel` 将专家参数按 dim=0 分片到 EP mesh
- **Context Parallel (CP)**：Ulysses-style 序列切分，通过 NPU CP registry 应用
- **Pipeline Parallel (PP)**：结构原生支持 `pipeline_llm`（ModuleDict layers + tok_embeddings/norm/output）
- **FSDP**：使用 `apply_fsdp`（来自 llama4）对非专家参数做 Fully Sharded Data Parallel
- **Activation Checkpointing**：支持 `full` / `selective` 模式
- **torch.compile**：细粒度编译，排除 experts 和 NPURMSNorm

### 3. `state_dict_adapter.py` — Checkpoint 转换

实现 HuggingFace ↔ torchtitan 格式的双向 state dict 转换，支持 w13 融合/拆分。

### 4. `config_registry.py` — 训练配置

| 配置名 | 用途 | 参数量 | NPU 数 |
|--------|------|--------|--------|
| `longcat_flash_smoketest` | 快速验证 | ~34M | 1 |
| `longcat_flash_debug` | 调试训练 | ~6.5B | 4+ |
| `longcat_flash_alpaca_8npu` | Alpaca 训练 | 12.5B | 8 |

### 5. `__init__.py` — 模型注册

定义模型 flavors（smoketest / full / debug / debug_8npu）和 `model_registry()` 入口。

## 训练启动方式

### 方式一：Docker 容器启动（推荐）

```bash
# 启动训练容器（交互模式）
MODE=interactive bash scripts/docker_run_longcat_flash.sh

# 容器内执行训练
NGPU=8 CONFIG=longcat_flash_alpaca_8npu bash scripts/run_longcat_flash.sh
```

### 方式二：直接在容器内运行

```bash
# 启动持久化容器
bash scripts/docker_run_longcat_flash.sh

# 进入容器执行
docker exec -it longcat-flash-train bash

# 安装（首次）
pip install -e . --no-deps

# 8卡训练
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
HCCL_CONNECT_TIMEOUT=3600 \
TASK_QUEUE_ENABLE=2 \
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 \
--local-ranks-filter 0 --role rank --tee 3 \
-m torchtitan_npu.entry \
--module torchtitan_npu.models.longcat_flash \
--config longcat_flash_alpaca_8npu
```

### 方式三：单卡 Smoketest

```bash
WORLD_SIZE=1 RANK=0 LOCAL_RANK=0 MASTER_ADDR=127.0.0.1 MASTER_PORT=29500 \
python3 -m torchtitan_npu.entry \
--module torchtitan_npu.models.longcat_flash \
--config longcat_flash_smoketest
```

### 训练参数覆盖

```bash
--training.steps=50
--training.seq_len=1024
--parallelism.context_parallel_degree=4
--parallelism.pipeline_parallel_degree=2
```

## 实验结果

### 硬件环境

- 8× Ascend 910 (64GB HBM each, peak FLOPS 354 TFLOPS)
- Docker: `torchtitan-npu:cann9.0.0-torch2.12.0`
- PyTorch 2.12.0 + torch_npu 2.12.0rc1
- CANN 9.0.0

### 实验 1：基线（无优化, 16 experts）

**模型配置 (`debug` flavor, 早期版本)：**
- dim=6144, num_layers=4, num_routed_experts=16, num_zero_experts=8
- moe_top_k=12, expert_ffn_hidden_dim=2048
- 总参数量: 6.56B

**训练配置：**
- local_batch_size=2, seq_len=2048
- optimizer: AdamW (lr=2e-5, swap_optimizer=True)
- activation_checkpoint: full
- parallelism: EP=0, FSDP=8 (纯 FSDP)

**结果 (10 步)：**
| 指标 | 值 |
|------|-----|
| Loss | 12.23 → 8.43 |
| 内存/卡 | 39.89 GiB (65.1%) |
| 速度 | 2.05s/step |
| TFLOPS | 73 |
| MFU | **20.6%** |

**复现命令：**
```bash
# 需要回退到早期 commit 的 debug 配置 (16 experts, 4 layers, EP=0)
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 \
-m torchtitan_npu.entry --module torchtitan_npu.models.longcat_flash \
--config longcat_flash_alpaca_8npu --training.steps=10
```

---

### 实验 2：+w13 融合 +npu_rms_norm（128 experts, EP=8）

**模型配置 (`debug_8npu` flavor)：**
- dim=6144, num_layers=2, num_routed_experts=128, num_zero_experts=64
- moe_top_k=8, routed_scaling_factor=4.0, expert_ffn_hidden_dim=2048
- q_lora_rank=1536, kv_lora_rank=512, qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128
- 总参数量: 12.5B

**训练配置：**
- local_batch_size=2, seq_len=2048
- optimizer: AdamW (lr=2e-5, eps=1e-8, swap_optimizer=True, swap_optimizer_times=16)
- lr_scheduler: cosine (warmup=10, decay_ratio=0.9, min_lr_factor=0.1)
- activation_checkpoint: full
- fsdp_reshard_after_forward: always
- converters: [npu_rms_norm]

**并行配置：**
- EP=8, TP=1, PP=1, CP=1, FSDP=-1 (auto)

**优化特性：**
- w1+w3 融合为 w13 (2次 grouped_mm 替代 3次)
- npu_rms_norm converter 启用
- Python argsort+scatter_add token dispatch

**结果 (10 步)：**
| 指标 | 值 |
|------|-----|
| Loss | 12.24 → 8.30 |
| 内存/卡 | 38.74 GiB (63.2%) |
| 速度 | 2.65s/step |
| TFLOPS | 110 |
| MFU | **31.1%** |

**复现命令：**
```bash
# 禁用 npu_moe_token_permute (在 model.py 的 LongCatFlashMoE.forward 中)
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 \
-m torchtitan_npu.entry --module torchtitan_npu.models.longcat_flash \
--config longcat_flash_alpaca_8npu --training.steps=10
```

---

### 实验 3：+npu_moe_token_permute/unpermute（当前最优）

**模型配置：** 同实验 2

**训练配置：** 同实验 2

**优化特性（在实验 2 基础上新增）：**
- `torch_npu.npu_moe_token_permute` 替代 Python argsort
- `torch_npu.npu_moe_token_unpermute` 替代 Python scatter_add
- Token routing 完全下推到 NPU 硬件

**结果 (20 步, 多次运行稳态平均)：**
| 指标 | 值 |
|------|-----|
| Loss | 12.26 → 7.31 |
| 内存/卡 | 37.46–37.71 GiB (61.1–61.6%) |
| 速度 | 2.27–2.62s/step |
| TFLOPS | 112–129 |
| MFU | **31.5–36.5%** |

> MFU 在不同运行间有 ~5% 波动，与 NPU thermal state 和 HCCL 初始化有关。

**复现命令（当前代码）：**
```bash
PYTORCH_NPU_ALLOC_CONF=expandable_segments:True \
HCCL_CONNECT_TIMEOUT=3600 \
TASK_QUEUE_ENABLE=2 \
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 \
--local-ranks-filter 0 --role rank --tee 3 \
-m torchtitan_npu.entry \
--module torchtitan_npu.models.longcat_flash \
--config longcat_flash_alpaca_8npu \
--training.steps=20
```

---

### 实验 4：MLA Absorb 模式测试

**模型配置：** 同实验 2，`enable_mla_absorb=True`

**MLA Absorb 原理：**
- 将 `kv_b_proj.weight` 分解为 `w_uk (heads, qk_nope_head_dim, kv_lora_rank)` 和 `w_uv (heads, v_head_dim, kv_lora_rank)`
- Q 通过 einsum `"bhsq,hqr->bhsr"` 投影到 kv_lora_rank 维度
- Attention 在 latent space (kv_lora_rank+qk_rope_head_dim) 操作
- Output 通过 einsum `"bhsr,hrv->bhsv"` 还原到 v_head_dim

**结果 (10 步)：**
| 指标 | 值 |
|------|-----|
| Loss | 12.24 → 8.37 |
| 内存/卡 | 48.41 GiB (79.0%) |
| 速度 | 3.02s/step |
| TFLOPS | 97 |
| MFU | **27.4%** |

**结论：** MLA absorb 对 LongCat-Flash **不适用**。因为 `kv_lora_rank(512) > qk_nope_head_dim(128)`，absorb 后 attention head_dim 从 192 增大到 576，计算量反而增加 3 倍。仅当 `kv_lora_rank < qk_nope_head_dim` 时有收益。默认禁用。

**复现命令：**
```bash
# 修改 model.py 中 self.enable_mla_absorb = True
torchrun --nproc_per_node=8 --master_addr=127.0.0.1 --master_port=29500 \
-m torchtitan_npu.entry --module torchtitan_npu.models.longcat_flash \
--config longcat_flash_alpaca_8npu --training.steps=10
```

---

### 性能演进汇总

| 实验 | 优化 | 参数量 | 内存/卡 | 速度 | TFLOPS | MFU |
|------|------|--------|---------|------|--------|-----|
| 1 | 基线 | 6.56B | 39.89 GiB | 2.05s | 73 | 20.6% |
| 2 | +w13+npu_rms_norm | 12.5B | 38.74 GiB | 2.65s | 110 | 31.1% |
| 3 | +npu_moe_dispatch | 12.5B | 37.46 GiB | 2.27s | 129 | **36.5%** |
| 4 | +mla_absorb (负优化) | 12.5B | 48.41 GiB | 3.02s | 97 | 27.4% |

## 已实施的性能优化

### 1. 融合 w1+w3→w13 + npu_swiglu

将 gate_proj 和 up_proj 融合为单个 w13 参数，配合 `torch_npu.npu_swiglu` 做融合 SiLU+Gate 激活：
- 3 次 `grouped_mm` 减少为 2 次
- 减少 33% 的专家计算内存读取

### 2. NPU RMSNorm (`npu_rms_norm` converter)

使用 torchtitan 公共 `nn.RMSNorm`，通过 converter 自动替换为 `torch_npu.npu_rms_norm` 硬件加速算子。

### 3. NPU MoE Token Dispatch

使用 `torch_npu.npu_moe_token_permute` / `torch_npu.npu_moe_token_unpermute` 替代 Python 级别的 argsort + scatter_add。

### 4. Tensor Parallel for MLA

MLA 注意力张量并行：q_b_proj/kv_b_proj ColwiseParallel, o_proj RowwiseParallel, FFN ColwiseParallel/RowwiseParallel, norms SequenceParallel。

### 5. Context Parallel (CP)

Ulysses-style 序列切分，通过 `apply_cp_to_attention_module` 注册。要求 `num_heads % cp_degree == 0`。

### 6. Pipeline Parallel (PP)

结构原生支持（ModuleDict layers），通过 `pipeline_llm` 自动切分。

### 7. MLA Absorb（默认禁用）

Weight absorption 将 kv_b_proj 分解吸收到 Q 和 output。仅当 `kv_lora_rank < qk_nope_head_dim` 时有收益。

### 8. torch.compile（选择性编译）

细粒度编译排除 experts (grouped_mm) 和 NPURMSNorm。通过 `compile_config.enable=True` 激活。

## 未来可优化方向

### 1. NPU RoPE (`npu_rotary_mul`)

**状态：** 已调查，CANN 9.0.0 不兼容

`torch_npu.npu_rotary_mul` 的 forward 在所有 `qk_rope_head_dim ≤ 64` 下正常工作，但 **backward 在 activation checkpointing recomputation 期间失败**（`aclnnRotaryPositionEmbeddingV2` 异步错误）。

**根因：** CANN 9.0.0 的 rotary backward kernel 与 PyTorch AC dispatcher 交互时触发内部错误。

**解决方案：** 等待 CANN 升级修复，或实现自定义 `torch.autograd.Function` 避免 AC 对 rotary backward 的重计算。

### 2. NpuExpertParallel

**状态：** 已调查，架构不兼容，需完整重构

`NpuExpertParallel` 设计为替代整个 token dispatch 流程（通过 `_token_dispatch` / `_token_combine` hooks），但 LongCat-Flash 的 MoE 有独特的 **zero-expert identity routing** 且已在 MoE forward 中做了本地 `npu_moe_token_permute`。

**不兼容的核心矛盾：**

| 维度 | LongCat-Flash 当前实现 | NpuExpertParallel 期望 |
|------|---|---|
| Token dispatch 位置 | MoE.forward 内部做 `npu_moe_token_permute` | EP input hook 做全局 all-to-all + `aclnnMoeReRouting` |
| num_tokens_per_expert | 本地专家数 shape=(16,) | 全局专家数 shape=(128,) 覆盖所有 EP ranks |
| Expert forward 输入 | 已 permute 好的 local tokens | 原始 routed tokens（EP hook 再次 permute） |
| Routing scores | 在 `npu_moe_token_unpermute` 的 `probs` 参数中应用 | 作为 3rd arg 传入 experts，EP hook 对其做 all-to-all |

**解决方案（需独立 PR）：**
1. 将 `LongCatFlashMoE` 重构为继承 `torchtitan.models.common.moe.MoE`
2. 使用标准 `TokenChoiceTopKRouter` + `TokenReorderer` 做路由
3. 将 zero-expert identity 逻辑实现为 `shared_experts` 或 post-processing hook
4. 移除 MoE forward 中的 `npu_moe_token_permute` 调用（让 `NpuExpertParallel` 接管）
5. 确保 experts forward 签名为 `(x, num_tokens_per_expert, routed_scores=None)`

### 3. 512 专家全量支持

需 16+ 卡。512 experts × 2 layers 的总参数量为 ~80GB (BF16)，8×64GB HBM 在 EP=8 时仍然 OOM（单卡需持有 64 experts ≈ 9.7GB 权重 + 梯度 + activations）。

方案：EP=16 (16 卡) 或 EP=8+FSDP=2 (16 卡) 或 CPU offload pipeline。

### 4. 量化 GMM (`npu_quant_gmm`)

支持 MXFP8 / HiFloat8 精度的 grouped matmul，进一步减少专家计算的内存和算力消耗。
