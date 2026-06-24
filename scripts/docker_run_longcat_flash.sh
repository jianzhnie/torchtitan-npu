#!/bin/bash
# Launch LongCat-Flash-Chat training inside the torchtitan-npu Docker container.
#
# Usage:
#   # Smoketest (1 NPU)
#   NGPU=1 CONFIG=longcat_flash_smoketest bash scripts/docker_run_longcat_flash.sh
#
#   # Alpaca training (8 NPUs, default)
#   bash scripts/docker_run_longcat_flash.sh
#
#   # Interactive shell inside the container
#   MODE=interactive bash scripts/docker_run_longcat_flash.sh

set -euo pipefail

DOCKER_IMAGE=${DOCKER_IMAGE:-"torchtitan-npu:cann9.0.0-torch2.12.0"}
CONTAINER_NAME=${CONTAINER_NAME:-"longcat-flash-train"}
NGPU=${NGPU:-"8"}
CONFIG=${CONFIG:-"longcat_flash_alpaca_8npu"}
MODE=${MODE:-"train"}

REPO_DIR="/home/jianzhnie/llmtuner/llm/torchtitan-npu"
MODEL_DIR="/home/jianzhnie/llmtuner/hfhub/models/meituan-longcat/LongCat-Flash-Chat"
DATASET_DIR="/home/jianzhnie/llmtuner/hfhub/datasets/tatsu-lab/alpaca"

if [[ -n "$(docker ps -aq -f name="^/${CONTAINER_NAME}$")" ]]; then
    echo "Container '${CONTAINER_NAME}' already exists. Removing it..."
    docker rm -f "${CONTAINER_NAME}"
fi

DOCKER_COMMON_ARGS=(
    -u root
    --name "${CONTAINER_NAME}"
    --ipc=host
    --net=host
    --ulimit memlock=-1
    --ulimit stack=67108864
    --privileged=true
    --device=/dev/davinci0
    --device=/dev/davinci1
    --device=/dev/davinci2
    --device=/dev/davinci3
    --device=/dev/davinci4
    --device=/dev/davinci5
    --device=/dev/davinci6
    --device=/dev/davinci7
    --device=/dev/davinci_manager
    --device=/dev/devmm_svm
    --device=/dev/hisi_hdc
    --shm-size=256g
    -e HCCL_BUFFSIZE=1024
    -e HCCL_BUFFER_FILE_SIZE=1024
    -e NGPU="${NGPU}"
    -e CONFIG="${CONFIG}"
    -e LOG_RANK="${LOG_RANK:-0}"
    -v /usr/local/dcmi:/usr/local/dcmi
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver
    -v /usr/local/Ascend/add-ons/:/usr/local/Ascend/add-ons/
    -v /usr/local/Ascend/driver/tools/hccn_tool:/usr/local/Ascend/driver/tools/hccn_tool
    -v /usr/local/Ascend/driver/lib64/:/usr/local/Ascend/driver/lib64/
    -v /usr/local/Ascend/driver/version.info:/usr/local/Ascend/driver/version.info
    -v /etc/ascend_install.info:/etc/ascend_install.info
    -v /home/jianzhnie/llmtuner:/home/jianzhnie/llmtuner:rw
    -w "${REPO_DIR}"
)

INSTALL_CMD="pip install -e . --no-deps 2>&1 | tail -1"

if [ "${MODE}" = "interactive" ]; then
    echo "Launching interactive shell in ${DOCKER_IMAGE} ..."
    docker run -it "${DOCKER_COMMON_ARGS[@]}" "${DOCKER_IMAGE}" \
        bash -c "${INSTALL_CMD} && exec bash"
else
    echo "Training LongCat-Flash-Chat: NGPU=${NGPU}, CONFIG=${CONFIG}"
    docker run -it "${DOCKER_COMMON_ARGS[@]}" "${DOCKER_IMAGE}" \
        bash -c "${INSTALL_CMD} && bash scripts/run_longcat_flash.sh"
fi
