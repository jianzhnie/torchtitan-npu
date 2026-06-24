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
- **`LongCatFlashMLA`** — Multi-head Latent Attention，含 q/kv 的 LoRA 压缩、RoPE 和 SDPA
- **`LongCatFlashMoE`** — 混合专家层，实现 softmax routing + e_score_correction_bias + identity zero-expert
- **`LongCatFlashExperts`** — 专家计算，使用 **w13 融合**（gate_proj + up_proj 合并为单参数）+ `grouped_mm` + `npu_swiglu`

**性能优化要点：**
- w1+w3 融合为 w13：3 次 `grouped_mm` 减少为 2 次
- 使用 `torch_npu.npu_swiglu` 替代手动 SiLU+Multiply
- 使用 torchtitan 公共 `RMSNorm`（继承 `nn.RMSNorm`），支持 `npu_rms_norm` converter 自动替换为硬件加速算子
- EP 模式下通过 `_to_local()` 处理 DTensor，兼容 Expert Parallel

### 2. `parallelize.py` — 并行化策略

- **Expert Parallel (EP)**：通过 `ExpertParallel` 将专家参数按 dim=0 分片到 EP mesh
- **FSDP**：使用 `apply_fsdp`（来自 llama4）对非专家参数做 Fully Sharded Data Parallel
- **Activation Checkpointing**：支持 `full` / `selective` 模式
- **torch.compile**：支持每层编译加速

### 3. `state_dict_adapter.py` — Checkpoint 转换

实现 HuggingFace ↔ torchtitan 格式的双向 state dict 转换：

- Embedding: `model.embed_tokens.*` ↔ `tok_embeddings.*`
- Layers: `model.layers.{L}.*` ↔ `layers.{L}.*`
- Router: `mlp.router.classifier.*` ↔ `moe.gate.*`
- Experts: HF 的 per-expert `gate_proj.weight` + `up_proj.weight` ↔ titan 的融合 `w13:expert_{N}`
- Output: `lm_head.*` ↔ `output.*`

### 4. `config_registry.py` — 训练配置

提供三个预定义训练配置：

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

可通过命令行覆盖配置参数：

```bash
# 修改训练步数
--training.steps=50

# 修改序列长度
--training.seq_len=1024
```

### 当前验证结果

8×Ascend 910 (64GB HBM) 上的训练结果：

| 指标 | 值 |
|------|-----|
| 模型 | debug_8npu (128 experts, 2 layers) |
| 参数量 | 12.5B |
| 并行策略 | EP=8 + FSDP + AC(full) |
| Loss (20步) | 12.35 → 7.31 |
| 内存/卡 | 38.78 GiB (63.3%) |
| 速度 | ~2.65s/step |
| TFLOPS | 110 |
| MFU | 31% |

## 未来可优化方向

### 高优先级

1. **Tensor Parallel for MLA**
   - 当前 TP 未实现（`NotImplementedError`）
   - 参考 `deepseek_v32/parallelize.py` 的 `apply_non_moe_tp()` 实现 MLA 的 TP plan
   - 需要对 `q_a_proj`、`kv_a_proj_with_mqa`、`kv_b_proj`、`o_proj` 分别配置 ColwiseParallel/RowwiseParallel
   - 预期收益：单卡内存减半，支持更大模型

2. **NPU MoE Dispatch (`npu_moe_dispatch`)**
   - 当前使用 Python 级别的 argsort + scatter_add 做 token routing
   - 可替换为 `torch_npu.npu_moe_token_permute` / `torch_npu.npu_moe_token_unpermute`
   - 需要让 `LongCatFlashMoE` 兼容 `NpuExpertParallel`（EP 的 rerouting 用 `npu_moe_re_routing`）
   - 预期收益：token dispatch 速度提升 2-3x

3. **NPU RoPE (`npu_rope`)**
   - 当前使用 Python 实现的 `apply_rotary_pos_emb_mla`
   - MLA 的 RoPE 采用 interleaved 排列后 apply cos/sin，可适配 `torch_npu.npu_rotary_mul`
   - 需要调整 reshape 顺序使其符合 `npu_rotary_mul` 的输入格式
   - 预期收益：RoPE 计算加速 ~50%

### 中优先级

4. **torch.compile 细粒度编译**
   - 参考 deepseek_v32 的 `apply_compile()`：对 MoE 子模块、attention 子模块分别编译
   - 排除 experts（grouped_mm 不支持 dynamo）和 NPURMSNorm
   - 预期收益：非 MoE 部分的 kernel fusion 提升 10-20%

5. **MLA Absorb 优化**
   - 参考 deepseek_v32 的 `enable_mla_absorb` 模式
   - 将 `kv_b_proj` 权重分解为 `w_uk` 和 `w_uv`，通过 einsum 吸收到 Q 和 output
   - 减少 KV cache 大小和 attention 计算量
   - 预期收益：attention 显存减少 ~40%

6. **512 专家全量支持**
   - 当前 8 卡最多训练 128 专家（受 HBM 限制）
   - 512 专家需要 16+ 卡（EP=16）或 EP=8 + FSDP=2（16 卡）
   - 或实现 CPU offload + prefetch 的流水线方案

### 低优先级

7. **量化 GMM (`npu_quant_gmm`)**
   - 支持 MXFP8 / HiFloat8 精度的 grouped matmul
   - 进一步减少专家计算的内存和算力消耗

8. **Context Parallel**
   - 支持超长序列训练（>32K）
   - 需要在 attention 层引入 Ulysses 或 Ring 风格的 CP

9. **Pipeline Parallel**
   - 对完整 28 层模型做 PP 切分
   - 配合 `pipeline_llm` pipelining_fn 使用
