from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torchtitan.models.common.embedding import Embedding
from torchtitan.models.common.linear import Linear
from torchtitan.models.common.rmsnorm import RMSNorm
from torchtitan.models.utils import get_dense_model_nparams_and_flops
from torchtitan.protocols.model import BaseModel
from torchtitan.protocols.module import Module, ModuleDict, ModuleList

logger = logging.getLogger()


def _to_local(param: torch.Tensor) -> torch.Tensor:
    """Extract the local tensor from a DTensor, or return as-is for plain tensors."""
    if hasattr(param, "to_local"):
        return param.to_local()
    return param


def precompute_freqs_cis(
    dim: int,
    max_seq_len: int,
    theta: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    positions = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(positions, inv_freq)
    cos = torch.cat([freqs.cos(), freqs.cos()], dim=-1)
    sin = torch.cat([freqs.sin(), freqs.sin()], dim=-1)
    return cos, sin


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb_mla(
    q_rope: torch.Tensor,
    k_rope: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    b, h, s, d = q_rope.shape
    q_rope = q_rope.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    b, h, s, d = k_rope.shape
    k_rope = k_rope.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    q_embed = q_rope * cos + rotate_half(q_rope) * sin
    k_embed = k_rope * cos + rotate_half(k_rope) * sin
    return q_embed, k_embed



class LongCatFlashMLA(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        model_args: LongCatFlashModel.Config

    def __init__(self, config: Config):
        super().__init__()
        args = config.model_args
        self.num_heads = args.num_heads
        self.q_lora_rank = args.q_lora_rank
        self.kv_lora_rank = args.kv_lora_rank
        self.qk_nope_head_dim = args.qk_nope_head_dim
        self.qk_rope_head_dim = args.qk_rope_head_dim
        self.v_head_dim = args.v_head_dim
        qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim

        self.q_a_proj = Linear.Config(
            in_features=args.dim, out_features=args.q_lora_rank, bias=False,
        ).build()
        self.q_a_layernorm = RMSNorm.Config(normalized_shape=args.q_lora_rank, eps=args.norm_eps).build()
        self.q_b_proj = Linear.Config(
            in_features=args.q_lora_rank, out_features=self.num_heads * qk_head_dim, bias=False,
        ).build()

        self.kv_a_proj_with_mqa = Linear.Config(
            in_features=args.dim,
            out_features=self.kv_lora_rank + self.qk_rope_head_dim,
            bias=False,
        ).build()
        self.kv_a_layernorm = RMSNorm.Config(normalized_shape=self.kv_lora_rank, eps=args.norm_eps).build()
        self.kv_b_proj = Linear.Config(
            in_features=self.kv_lora_rank,
            out_features=self.num_heads * (self.qk_nope_head_dim + self.v_head_dim),
            bias=False,
        ).build()

        self.o_proj = Linear.Config(
            in_features=self.num_heads * self.v_head_dim,
            out_features=args.dim,
            bias=False,
        ).build()

        self.mla_scale_q_lora: float | None = None
        self.mla_scale_kv_lora: float | None = None
        if args.mla_scale_q_lora:
            self.mla_scale_q_lora = (args.dim / args.q_lora_rank) ** 0.5
        if args.mla_scale_kv_lora:
            self.mla_scale_kv_lora = (args.dim / args.kv_lora_rank) ** 0.5

        self.scaling = qk_head_dim ** (-0.5)
        self.enable_mla_absorb = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seq_length, _ = hidden_states.shape

        q_states = self.q_b_proj(self.q_a_layernorm(self.q_a_proj(hidden_states)))
        qk_head_dim = self.qk_nope_head_dim + self.qk_rope_head_dim
        q_states = q_states.view(batch_size, seq_length, self.num_heads, qk_head_dim).transpose(1, 2)
        q_nope, q_rot = q_states.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)

        if self.mla_scale_q_lora is not None:
            q_nope = q_nope * self.mla_scale_q_lora
            q_rot = q_rot * self.mla_scale_q_lora

        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)
        kv_latent, k_rot = compressed_kv.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        kv_latent = self.kv_a_layernorm(kv_latent)

        if self.mla_scale_kv_lora is not None:
            kv_latent = kv_latent * self.mla_scale_kv_lora

        k_rot = k_rot.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        q_rot, k_rot = apply_rotary_pos_emb_mla(q_rot, k_rot, cos, sin)
        k_rot = k_rot.expand(-1, self.num_heads, -1, -1)

        if self.enable_mla_absorb:
            attn_output = self._forward_absorb(
                q_nope, q_rot, kv_latent, k_rot, batch_size, seq_length
            )
        else:
            attn_output = self._forward_standard(
                q_nope, q_rot, kv_latent, k_rot, batch_size, seq_length
            )

        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_length, -1)
        return self.o_proj(attn_output)

    def _forward_absorb(
        self, q_nope, q_rot, kv_latent, k_rot, batch_size, seq_length,
    ) -> torch.Tensor:
        """MLA with weight absorption — avoids expanding kv_b_proj to full heads."""
        wkv_b = self.kv_b_proj.weight.reshape(
            self.num_heads, self.qk_nope_head_dim + self.v_head_dim, self.kv_lora_rank
        )
        w_uk = wkv_b[:, :self.qk_nope_head_dim, :]
        w_uv = wkv_b[:, self.qk_nope_head_dim:, :]
        w_uv_t = w_uv.permute(0, 2, 1).contiguous()

        q_nope = torch.einsum(
            "bhsq,hqr->bhsr", q_nope, w_uk
        )

        k_nope = kv_latent.unsqueeze(1)
        v = kv_latent.unsqueeze(1)

        query_states = torch.cat([q_nope, q_rot], dim=-1)
        key_states = torch.cat([k_nope.expand(-1, self.num_heads, -1, -1), k_rot], dim=-1)

        absorb_scaling = (self.kv_lora_rank + self.qk_rope_head_dim) ** (-0.5)
        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, v.expand(-1, self.num_heads, -1, -1),
            is_causal=True, scale=absorb_scaling,
        )

        attn_output = torch.einsum("bhsr,hrv->bhsv", attn_output, w_uv_t)
        return attn_output

    def _forward_standard(
        self, q_nope, q_rot, kv_latent, k_rot, batch_size, seq_length,
    ) -> torch.Tensor:
        """Standard MLA — expands kv_b_proj fully."""
        k_pass = self.kv_b_proj(kv_latent)
        key_shape = (batch_size, seq_length, self.num_heads, self.qk_nope_head_dim + self.v_head_dim)
        k_pass = k_pass.view(key_shape).transpose(1, 2)
        k_pass, value_states = k_pass.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)

        query_states = torch.cat([q_nope, q_rot], dim=-1)
        key_states = torch.cat([k_pass, k_rot], dim=-1)

        attn_output = F.scaled_dot_product_attention(
            query_states, key_states, value_states,
            is_causal=True, scale=self.scaling,
        )
        return attn_output

class LongCatFlashExperts(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        dim: int
        hidden_dim: int
        num_experts: int

    def __init__(self, config: Config):
        super().__init__()
        self.num_experts = config.num_experts
        self.w13 = nn.Parameter(
            torch.empty(config.num_experts, config.hidden_dim * 2, config.dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(config.num_experts, config.dim, config.hidden_dim)
        )

    def forward(
        self, x_RD: torch.Tensor, num_tokens_per_expert: torch.Tensor,
        routed_scores: torch.Tensor | None = None,
    ) -> torch.Tensor:
        offsets = torch.cumsum(num_tokens_per_expert, dim=0, dtype=torch.int64)
        w13 = _to_local(self.w13)
        w2 = _to_local(self.w2)
        h = torch._grouped_mm(
            x_RD.bfloat16(), w13.bfloat16().transpose(-2, -1), offs=offsets
        )
        try:
            import torch_npu
            h = torch_npu.npu_swiglu(h, dim=-1)
        except ImportError:
            half = h.shape[-1] // 2
            h = F.silu(h[..., :half]) * h[..., half:]
        if routed_scores is not None:
            h = h * routed_scores.to(h.dtype)
        return torch._grouped_mm(
            h, w2.bfloat16().transpose(-2, -1), offs=offsets
        ).type_as(x_RD)


class LongCatFlashMoE(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        model_args: LongCatFlashModel.Config

    def __init__(self, config: Config):
        super().__init__()
        args = config.model_args
        self.num_real_experts = args.num_routed_experts
        self.num_zero_experts = args.num_zero_experts
        total_experts = self.num_real_experts + self.num_zero_experts
        self.top_k = args.moe_top_k
        self.routed_scaling_factor = args.routed_scaling_factor
        self.load_balance_coeff: float | None = None

        self.gate = Linear.Config(
            in_features=args.dim, out_features=total_experts, bias=False,
        ).build()
        self.register_buffer(
            "e_score_correction_bias",
            torch.zeros(total_experts, dtype=torch.float32),
        )

        self.experts = LongCatFlashExperts.Config(
            dim=args.dim,
            hidden_dim=args.expert_ffn_hidden_dim,
            num_experts=self.num_real_experts,
        ).build()

        self.register_buffer(
            "tokens_per_expert",
            torch.zeros(self.num_real_experts, dtype=torch.float32),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        orig_shape = x.shape
        _bs, _slen, dim = x.shape
        x_flat = x.view(-1, dim)

        scores = self.gate(x_flat.float())
        scores = F.softmax(scores, dim=-1)
        scores_for_choice = scores + self.e_score_correction_bias.unsqueeze(0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False).indices
        topk_weights = scores.gather(1, topk_indices)
        topk_weights = topk_weights * self.routed_scaling_factor

        identity_mask = topk_indices >= self.num_real_experts
        identity_weight = (topk_weights * identity_mask.float()).sum(dim=-1, keepdim=True)
        identity_out = x_flat * identity_weight

        real_weights = topk_weights.clone()
        real_weights[identity_mask] = 0.0
        real_ids = topk_indices.clone()
        real_ids[identity_mask] = 0

        try:
            import torch_npu
            routed_input, sorted_indices = torch_npu.npu_moe_token_permute(
                x_flat.bfloat16(), real_ids.to(torch.int64)
            )

            num_tokens_per_expert = torch.histc(
                real_ids.float(),
                bins=self.num_real_experts,
                min=0,
                max=self.num_real_experts - 1,
            ).int()

            with torch.no_grad():
                self.tokens_per_expert.add_(num_tokens_per_expert.float())

            routed_output = self.experts(
                routed_input, num_tokens_per_expert
            )
            unpermuted = torch_npu.npu_moe_token_unpermute(
                routed_output.bfloat16(), sorted_indices,
                probs=real_weights.bfloat16(),
            )
            expert_out = unpermuted.to(x_flat.dtype)
        except ImportError:
            expert_out = self._dispatch_python(x_flat, real_ids, real_weights)

        out = expert_out + identity_out
        return out.view(orig_shape)

    def _dispatch_python(
        self, x_flat: torch.Tensor, real_ids: torch.Tensor, real_weights: torch.Tensor,
    ) -> torch.Tensor:
        """Fallback Python dispatch when torch_npu is not available."""
        dim = x_flat.shape[-1]
        expert_ids_flat = real_ids.view(-1)
        scores_flat = real_weights.view(-1)
        sorted_perm = expert_ids_flat.argsort(stable=True)
        sorted_scores = scores_flat[sorted_perm]
        token_indices = sorted_perm // self.top_k

        num_tokens_per_expert = torch.histc(
            expert_ids_flat[sorted_perm].float(),
            bins=self.num_real_experts,
            min=0,
            max=self.num_real_experts - 1,
        ).int()

        with torch.no_grad():
            self.tokens_per_expert.add_(num_tokens_per_expert.float())

        routed_input = x_flat[token_indices]
        routed_output = self.experts(routed_input, num_tokens_per_expert)
        routed_output = (
            routed_output.float() * sorted_scores.unsqueeze(-1)
        ).to(x_flat.dtype)

        out = torch.zeros_like(x_flat)
        out.scatter_add_(
            0,
            token_indices.unsqueeze(-1).expand(-1, dim),
            routed_output,
        )
        return out
        return out.view(orig_shape)


class LongCatFlashFeedForward(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        model_args: LongCatFlashModel.Config

    def __init__(self, config: Config):
        super().__init__()
        args = config.model_args
        self.gate_proj = Linear.Config(
            in_features=args.dim, out_features=args.ffn_hidden_dim, bias=False,
        ).build()
        self.up_proj = Linear.Config(
            in_features=args.dim, out_features=args.ffn_hidden_dim, bias=False,
        ).build()
        self.down_proj = Linear.Config(
            in_features=args.ffn_hidden_dim, out_features=args.dim, bias=False,
        ).build()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LongCatFlashMacroBlock(Module):
    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        layer_idx: int
        model_args: LongCatFlashModel.Config

    def __init__(self, config: Config):
        super().__init__()
        args = config.model_args
        layer_idx = config.layer_idx
        self.layer_idx = layer_idx
        self.moe_enabled = True
        self.weight_init_std = 0.02 / (2 * (layer_idx + 1)) ** 0.5

        self.self_attn = ModuleList([
            LongCatFlashMLA.Config(layer_idx=layer_idx * 2 + i, model_args=args).build()
            for i in range(2)
        ])
        self.mlps = ModuleList([
            LongCatFlashFeedForward.Config(model_args=args).build()
            for _ in range(2)
        ])
        self.input_layernorm = ModuleList([
            RMSNorm.Config(normalized_shape=args.dim, eps=args.norm_eps).build()
            for _ in range(2)
        ])
        self.post_attention_layernorm = ModuleList([
            RMSNorm.Config(normalized_shape=args.dim, eps=args.norm_eps).build()
            for _ in range(2)
        ])
        self.moe = LongCatFlashMoE.Config(model_args=args).build()

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        for i in range(2):
            residual = hidden_states
            hidden_states = self.input_layernorm[i](hidden_states)
            hidden_states = self.self_attn[i](hidden_states, cos, sin)
            hidden_states = residual + hidden_states

            residual = hidden_states
            hidden_states = self.post_attention_layernorm[i](hidden_states)

            if i == 0:
                shortcut_moe_output = self.moe(hidden_states)

            hidden_states = self.mlps[i](hidden_states)
            hidden_states = residual + hidden_states

            if i == 1:
                hidden_states = hidden_states + shortcut_moe_output

        return hidden_states

    def init_weights(self, buffer_device: torch.device):
        for norm in (*self.input_layernorm, *self.post_attention_layernorm):
            nn.init.trunc_normal_(norm.weight, mean=1, std=0.02)
        for attn in self.self_attn:
            for param in attn.parameters():
                nn.init.trunc_normal_(param, mean=0.0, std=self.weight_init_std)
        for mlp in self.mlps:
            for param in mlp.parameters():
                nn.init.trunc_normal_(param, mean=0.0, std=self.weight_init_std)
        nn.init.trunc_normal_(self.moe.gate.weight, mean=0.0, std=self.weight_init_std)
        for name in ("w13", "w2"):
            param = getattr(self.moe.experts, name, None)
            if param is not None:
                nn.init.trunc_normal_(param, mean=0.0, std=self.weight_init_std)


class LongCatFlashModel(BaseModel):
    @dataclass(kw_only=True, slots=True)
    class Config(BaseModel.Config):
        dim: int = 6144
        vocab_size: int = 131072
        num_layers: int = 28
        num_heads: int = 64
        ffn_hidden_dim: int = 12288
        expert_ffn_hidden_dim: int = 2048
        num_routed_experts: int = 512
        num_zero_experts: int = 256
        moe_top_k: int = 12
        routed_scaling_factor: float = 6.0
        q_lora_rank: int = 1536
        kv_lora_rank: int = 512
        qk_nope_head_dim: int = 128
        qk_rope_head_dim: int = 64
        v_head_dim: int = 128
        rope_theta: float = 1e7
        norm_eps: float = 1e-5
        max_seq_len: int = 131072
        mla_scale_q_lora: bool = True
        mla_scale_kv_lora: bool = True

        @property
        def layers(self):
            return range(self.num_layers)

        def update_from_config(self, *, trainer_config, **kwargs) -> None:
            seq_len = trainer_config.training.seq_len
            if seq_len > self.max_seq_len:
                logger.warning(
                    f"Sequence length {seq_len} exceeds original maximum {self.max_seq_len}."
                )
            self.max_seq_len = seq_len

        def get_nparams_and_flops(
            self, model: nn.Module, seq_len: int,
        ) -> tuple[int, int]:
            return get_dense_model_nparams_and_flops(
                model=model,
                n_layers=self.num_layers * 2,
                n_heads=self.num_heads,
                head_dims=self.qk_nope_head_dim + self.qk_rope_head_dim + self.v_head_dim,
                seq_len=seq_len,
            )

    def __init__(self, config: LongCatFlashModel.Config):
        super().__init__()
        self.model_args = config
        self.tok_embeddings = Embedding.Config(
            num_embeddings=config.vocab_size,
            embedding_dim=config.dim,
        ).build()
        self.layers = ModuleDict()
        for layer_idx in range(config.num_layers):
            self.layers[str(layer_idx)] = LongCatFlashMacroBlock.Config(
                layer_idx=layer_idx, model_args=config,
            ).build()
        self.norm = RMSNorm.Config(normalized_shape=config.dim, eps=config.norm_eps).build()
        self.output = Linear.Config(
            in_features=config.dim, out_features=config.vocab_size, bias=False,
        ).build()
        cos, sin = precompute_freqs_cis(config.qk_rope_head_dim, config.max_seq_len, config.rope_theta)
        self.register_buffer("cos_cached", cos, persistent=False)
        self.register_buffer("sin_cached", sin, persistent=False)

    def forward(
        self,
        tokens: torch.Tensor,
        positions: torch.Tensor | None = None,
        attention_masks=None,
    ) -> torch.Tensor:
        seq_len = tokens.shape[1]
        h = self.tok_embeddings(tokens)
        cos = self.cos_cached[:seq_len].to(h.device)
        sin = self.sin_cached[:seq_len].to(h.device)
        for layer in self.layers.values():
            h = layer(h, cos, sin)
        h = self.norm(h)
        return self.output(h)

    def init_weights(self, buffer_device: torch.device | None = None) -> None:
        buffer_device = buffer_device or self.cos_cached.device
        cos, sin = precompute_freqs_cis(
            self.model_args.qk_rope_head_dim,
            self.model_args.max_seq_len,
            self.model_args.rope_theta,
        )
        with torch.device(buffer_device):
            self.cos_cached = cos.to(device=buffer_device)
            self.sin_cached = sin.to(device=buffer_device)
        if self.tok_embeddings is not None:
            nn.init.normal_(self.tok_embeddings.weight)
        for layer in self.layers.values():
            layer.init_weights(buffer_device=buffer_device)
        if self.norm is not None:
            nn.init.trunc_normal_(self.norm.weight, mean=1, std=0.02)
        final_out_std = self.model_args.dim ** -0.5
        if self.output is not None:
            nn.init.trunc_normal_(
                self.output.weight, mean=0.0, std=final_out_std,
                a=-3 * final_out_std, b=3 * final_out_std,
            )
