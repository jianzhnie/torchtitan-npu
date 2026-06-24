from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor.parallel import (
    ColwiseParallel,
    RowwiseParallel,
    SequenceParallel,
    parallelize_module,
)
from torchtitan.config import (
    ActivationCheckpointConfig,
    CompileConfig,
    ParallelismConfig,
    TrainingConfig,
)
from torchtitan.distributed import ParallelDims
from torchtitan.distributed.activation_checkpoint import apply_ac
from torchtitan.distributed.expert_parallel import ExpertParallel
from torchtitan.distributed.tensor_parallel import NoParallel
from torchtitan.distributed.utils import TORCH_DTYPE_MAP
from torchtitan.models.llama4.parallelize import apply_fsdp
from torchtitan.protocols import ModelConvertersContainer
from torchtitan.tools.logging import logger

from .model import LongCatFlashModel


def parallelize_longcat_flash(
    model: LongCatFlashModel,
    *,
    parallel_dims: ParallelDims,
    training: TrainingConfig,
    model_converters: ModelConvertersContainer.Config,
    parallelism: ParallelismConfig,
    compile_config: CompileConfig,
    ac_config: ActivationCheckpointConfig,
    dump_folder: str,
):
    if parallel_dims.tp_enabled:
        tp_mesh = parallel_dims.get_mesh("tp")
        _apply_tp(model, tp_mesh, parallelism)

    if parallel_dims.ep_enabled:
        _apply_expert_parallel(model, parallel_dims)

    apply_ac(model, ac_config)

    if compile_config.enable:
        for layer in model.layers.values():
            layer.compile(**compile_config.torch_compile_kwargs)

    if parallel_dims.fsdp_enabled or parallel_dims.ep_enabled:
        dp_mesh_names = (
            ["dp_replicate", "fsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["fsdp"]
        )
        dp_mesh = parallel_dims.get_mesh(dp_mesh_names)

        edp_mesh_names = (
            ["dp_replicate", "efsdp"]
            if parallel_dims.dp_replicate_enabled
            else ["efsdp"]
        )
        edp_mesh = parallel_dims.get_optional_mesh(edp_mesh_names)

        apply_fsdp(
            model,
            dp_mesh,
            param_dtype=TORCH_DTYPE_MAP[training.mixed_precision_param],
            reduce_dtype=TORCH_DTYPE_MAP[training.mixed_precision_reduce],
            pp_enabled=parallel_dims.pp_enabled,
            cpu_offload=training.enable_cpu_offload,
            reshard_after_forward_policy=parallelism.fsdp_reshard_after_forward,
            ep_degree=parallel_dims.ep,
            edp_mesh=edp_mesh,
            gradient_divide_factor=parallel_dims.fsdp_gradient_divide_factor,
        )

        if parallel_dims.dp_replicate_enabled:
            logger.info("Applied HSDP to the model")
        else:
            logger.info("Applied FSDP to the model")

    return model


def _apply_tp(
    model: LongCatFlashModel,
    tp_mesh: DeviceMesh,
    parallelism: ParallelismConfig,
) -> None:
    """Apply Tensor Parallel to MLA attention projections and embeddings."""
    tp_degree = tp_mesh.size()

    model_plan = {
        "tok_embeddings": RowwiseParallel(),
        "norm": SequenceParallel(),
    }
    if not parallelism.disable_loss_parallel:
        model_plan["output"] = ColwiseParallel(
            input_layouts=None, output_layouts=None
        )
    else:
        model_plan["output"] = ColwiseParallel()

    parallelize_module(model, tp_mesh, model_plan)

    for layer in model.layers.values():
        for i in range(2):
            attn = layer.self_attn[i]
            num_heads = attn.num_heads
            if num_heads % tp_degree != 0:
                raise ValueError(
                    f"num_heads={num_heads} must be divisible by "
                    f"tensor_parallel_degree={tp_degree}."
                )
            attn.num_heads = num_heads // tp_degree

            attn_plan = {
                "q_a_proj": NoParallel(),
                "q_a_layernorm": NoParallel(),
                "q_b_proj": ColwiseParallel(use_local_output=True),
                "kv_a_proj_with_mqa": NoParallel(),
                "kv_a_layernorm": NoParallel(),
                "kv_b_proj": ColwiseParallel(use_local_output=True),
                "o_proj": RowwiseParallel(output_layouts=None),
            }
            parallelize_module(attn, tp_mesh, attn_plan)

        layer_plan = {
            f"input_layernorm.{i}": SequenceParallel()
            for i in range(2)
        }
        layer_plan.update({
            f"post_attention_layernorm.{i}": SequenceParallel()
            for i in range(2)
        })
        for i in range(2):
            layer_plan[f"mlps.{i}.gate_proj"] = ColwiseParallel()
            layer_plan[f"mlps.{i}.up_proj"] = ColwiseParallel()
            layer_plan[f"mlps.{i}.down_proj"] = RowwiseParallel(
                output_layouts=None
            )
        parallelize_module(layer, tp_mesh, layer_plan)

    logger.info("Applied Tensor Parallel (TP=%d) to MLA + FFN", tp_degree)


def _apply_expert_parallel(
    model: LongCatFlashModel,
    parallel_dims: ParallelDims,
) -> None:
    ep_mesh: DeviceMesh = parallel_dims.get_mesh("ep")
    for transformer_block in model.layers.values():
        if not transformer_block.moe_enabled:
            continue
        parallelize_module(
            module=transformer_block.moe.experts,
            device_mesh=ep_mesh,
            parallelize_plan=ExpertParallel(),
        )
    logger.info(
        "Applied Expert Parallel (EP=%d) to MoE experts", parallel_dims.ep
    )
