from torchtitan.components.loss import build_cross_entropy_loss
from torchtitan.components.optimizer import register_moe_load_balancing_hook
from torchtitan.distributed.pipeline_parallel import pipeline_llm
from torchtitan.protocols.model_spec import ModelSpec

from .model import LongCatFlashModel


def _make_longcat_flash_config(
    dim: int = 6144,
    vocab_size: int = 131072,
    num_layers: int = 28,
    num_heads: int = 64,
    ffn_hidden_dim: int = 12288,
    expert_ffn_hidden_dim: int = 2048,
    num_routed_experts: int = 512,
    num_zero_experts: int = 256,
    moe_top_k: int = 12,
    routed_scaling_factor: float = 6.0,
    q_lora_rank: int = 1536,
    kv_lora_rank: int = 512,
    qk_nope_head_dim: int = 128,
    qk_rope_head_dim: int = 64,
    v_head_dim: int = 128,
    rope_theta: float = 1e7,
    norm_eps: float = 1e-5,
    max_seq_len: int = 131072,
    mla_scale_q_lora: bool = True,
    mla_scale_kv_lora: bool = True,
) -> LongCatFlashModel.Config:
    return LongCatFlashModel.Config(
        dim=dim,
        vocab_size=vocab_size,
        num_layers=num_layers,
        num_heads=num_heads,
        ffn_hidden_dim=ffn_hidden_dim,
        expert_ffn_hidden_dim=expert_ffn_hidden_dim,
        num_routed_experts=num_routed_experts,
        num_zero_experts=num_zero_experts,
        moe_top_k=moe_top_k,
        routed_scaling_factor=routed_scaling_factor,
        q_lora_rank=q_lora_rank,
        kv_lora_rank=kv_lora_rank,
        qk_nope_head_dim=qk_nope_head_dim,
        qk_rope_head_dim=qk_rope_head_dim,
        v_head_dim=v_head_dim,
        rope_theta=rope_theta,
        norm_eps=norm_eps,
        max_seq_len=max_seq_len,
        mla_scale_q_lora=mla_scale_q_lora,
        mla_scale_kv_lora=mla_scale_kv_lora,
    )


def _smoketest_model() -> LongCatFlashModel.Config:
    return _make_longcat_flash_config(
        dim=128,
        vocab_size=131072,
        num_layers=2,
        num_heads=4,
        ffn_hidden_dim=256,
        expert_ffn_hidden_dim=64,
        num_routed_experts=8,
        num_zero_experts=4,
        moe_top_k=4,
        routed_scaling_factor=2.0,
        q_lora_rank=64,
        kv_lora_rank=32,
        qk_nope_head_dim=16,
        qk_rope_head_dim=8,
        v_head_dim=16,
        max_seq_len=2048,
    )


def _full_model() -> LongCatFlashModel.Config:
    return _make_longcat_flash_config()


def _debug_model() -> LongCatFlashModel.Config:
    return _make_longcat_flash_config(
        dim=6144,
        num_layers=2,
        num_routed_experts=512,
        num_zero_experts=256,
        max_seq_len=131072,
    )


def _debug_8npu_model() -> LongCatFlashModel.Config:
    return _make_longcat_flash_config(
        dim=6144,
        num_layers=2,
        num_routed_experts=128,
        num_zero_experts=64,
        moe_top_k=8,
        routed_scaling_factor=4.0,
        max_seq_len=4096,
    )


longcat_flash_configs = {
    "smoketest": _smoketest_model,
    "full": _full_model,
    "debug": _debug_model,
    "debug_8npu": _debug_8npu_model,
}


def model_registry(flavor: str) -> ModelSpec:
    from .parallelize import parallelize_longcat_flash
    from .state_dict_adapter import LongCatFlashStateDictAdapter

    return ModelSpec(
        name="longcat_flash",
        flavor=flavor,
        model=longcat_flash_configs[flavor](),
        parallelize_fn=parallelize_longcat_flash,
        pipelining_fn=pipeline_llm,
        build_loss_fn=build_cross_entropy_loss,
        post_optimizer_build_fn=register_moe_load_balancing_hook,
        state_dict_adapter=LongCatFlashStateDictAdapter,
    )
