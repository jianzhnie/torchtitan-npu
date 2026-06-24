#!/bin/bash
# Run LongCat-Flash-Chat training on Ascend NPUs.
#
# Usage:
#   # Smoketest (1 NPU, tiny model, 2 steps)
#   NGPU=1 CONFIG=longcat_flash_smoketest bash scripts/run_longcat_flash.sh
#
#   # Debug (8 NPUs, 4-layer model, 20 steps)
#   NGPU=8 CONFIG=longcat_flash_debug bash scripts/run_longcat_flash.sh
#
#   # Alpaca training (8 NPUs, 4-layer model, 100 steps)
#   NGPU=8 CONFIG=longcat_flash_alpaca_8npu bash scripts/run_longcat_flash.sh

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}
MODULE=${MODULE:-"torchtitan_npu.models.longcat_flash"}
CONFIG=${CONFIG:-"longcat_flash_alpaca_8npu"}
TRAIN_FILE=${TRAIN_FILE:-"torchtitan_npu.entry"}

PYTORCH_NPU_ALLOC_CONF="expandable_segments:True" \
CUDA_DEVICE_MAX_CONNECTIONS=1 \
CPU_AFFINITY_CONF=1 \
TASK_QUEUE_ENABLE=2 \
HCCL_CONNECT_TIMEOUT=3600 \
STREAMS_PER_DEVICE=32 \
MULTI_STREAM_MEMORY_RESERVE=1 \
torchrun --nproc_per_node=${NGPU} --master_addr=127.0.0.1 --master_port=29500 \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
-m ${TRAIN_FILE} --module ${MODULE} --config ${CONFIG} "$@"
