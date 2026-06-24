from torchtitan.components.lr_scheduler import LRSchedulersContainer
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.config import ActivationCheckpointConfig, DebugConfig
from torchtitan.hf_datasets.text_datasets import HuggingFaceTextDataLoader
from torchtitan.protocols.model_converter import ModelConvertersContainer

from torchtitan_npu.config.configs import (
    CheckpointConfig,
    OptimizerConfig,
    ParallelismConfig,
    ProfilingConfig,
    TrainerConfig,
    TrainingConfig,
)
from torchtitan_npu.converters import get_model_converter_config

from . import model_registry


_HF_ASSETS_PATH = (
    "/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat"
)
_ALPACA_DATASET_PATH = (
    "/home/jianzhnie/llmtuner/hfhub/datasets/tatsu-lab/alpaca"
)


def _default_converters():
    return [
        get_model_converter_config("npu_rms_norm"),
    ]


def longcat_flash_smoketest() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path=_HF_ASSETS_PATH,
        model_spec=model_registry("smoketest"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=2,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=128,
            max_norm=1.0,
            steps=2,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=1,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(enable=False),
        activation_checkpoint=ActivationCheckpointConfig(mode="selective"),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def longcat_flash_debug() -> TrainerConfig:
    return TrainerConfig(
        hf_assets_path=_HF_ASSETS_PATH,
        model_spec=model_registry("debug"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(dataset="c4_test"),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2.2e-4,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=4,
            decay_ratio=0.8,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            max_norm=1.0,
            steps=20,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=2,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="./checkpoints/LongCat-Flash-Chat",
            interval=500,
            last_save_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        profiling=ProfilingConfig(enable_profiling=False),
    )


def longcat_flash_alpaca_8npu() -> TrainerConfig:
    """Train LongCat-Flash-Chat (debug_8npu: 2 layers, 128 experts) on alpaca with 8 NPUs.

    Parallelism: EP=8 (each NPU holds 16 experts), no FSDP data sharding.
    Uses swap_optimizer to reduce HBM pressure.
    """
    return TrainerConfig(
        hf_assets_path=_HF_ASSETS_PATH,
        model_spec=model_registry("debug_8npu"),
        debug=DebugConfig(print_config=True),
        model_converters=ModelConvertersContainer.Config(converters=_default_converters()),
        metrics=MetricsProcessor.Config(log_freq=1),
        dataloader=HuggingFaceTextDataLoader.Config(
            dataset="alpaca",
            dataset_path=_ALPACA_DATASET_PATH,
        ),
        optimizer=OptimizerConfig(
            name="AdamW",
            lr=2e-5,
            eps=1e-8,
            swap_optimizer=True,
            swap_optimizer_times=16,
        ),
        lr_scheduler=LRSchedulersContainer.Config(
            warmup_steps=10,
            decay_ratio=0.9,
            decay_type="cosine",
            min_lr_factor=0.1,
        ),
        training=TrainingConfig(
            local_batch_size=2,
            seq_len=2048,
            max_norm=1.0,
            steps=100,
        ),
        parallelism=ParallelismConfig(
            data_parallel_replicate_degree=1,
            data_parallel_shard_degree=-1,
            fsdp_reshard_after_forward="always",
            tensor_parallel_degree=1,
            pipeline_parallel_degree=1,
            expert_parallel_degree=8,
            expert_tensor_parallel_degree=1,
            context_parallel_degree=1,
        ),
        checkpoint=CheckpointConfig(
            enable=False,
            folder="./checkpoints/LongCat-Flash-Chat-alpaca",
            interval=50,
            last_save_model_only=True,
        ),
        activation_checkpoint=ActivationCheckpointConfig(mode="full"),
        profiling=ProfilingConfig(enable_profiling=False),
    )
