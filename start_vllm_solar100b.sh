#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./start_vllm_solar100b.sh              # foreground (Ctrl+C to stop)
#   ./start_vllm_solar100b.sh start        # background detached container
#   ./start_vllm_solar100b.sh stop         # stop and remove the container
#   ./start_vllm_solar100b.sh build        # 로컬 Docker 이미지만 빌드 (재실행 시 생략됨)
# Optional env overrides:
#   MODEL_PATH, PORT, HOST, TP_SIZE, GPU_MEMORY_UTIL,
#   MAX_MODEL_LEN, SERVED_MODEL_NAME, CONTAINER_NAME, DOCKER_IMAGE, LOCAL_IMAGE

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_NAME="${CONTAINER_NAME:-solar-100b}"
DOCKER_IMAGE="${DOCKER_IMAGE:-upstage/vllm-solar-open:latest}"
# transformers 업그레이드가 적용된 로컬 이미지 (Dockerfile.solar 로 빌드)
LOCAL_IMAGE="${LOCAL_IMAGE:-solar-open-vllm:local}"
# MODEL_PATH를 지정하지 않으면 마운트된 HF 캐시에서 직접 로드합니다.
# HF repo ID(upstage/Solar-Open-100B) 대신 로컬 스냅샷 경로를 사용하면
# huggingface_hub의 cache 조회 메커니즘을 완전히 우회할 수 있습니다.
# 컨테이너 내부 경로 기준 (HOST: /home/common/huggingface → CONTAINER: /root/.cache/huggingface)
_SNAP="hub/models--upstage--Solar-Open-100B/snapshots/e9e68660442e3d962025ab72111e69e5a02b671d"
MODEL_PATH="${MODEL_PATH:-/root/.cache/huggingface/${_SNAP}}"

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
# Solar Open 100B: num_attention_heads=64 → TP size는 64의 약수여야 함 (1,2,4,8,...)
# 6×80GB GPU 중 4개(320GB)로 충분히 100B 모델을 수용, 나머지 2개는 유휴 상태
TP_SIZE="${TP_SIZE:-4}"
GPU_MEMORY_UTIL="${GPU_MEMORY_UTIL:-0.92}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-solar-100b}"

GPU_DEVICES="${GPU_DEVICES:-0,1,2,3}"
# HF 캐시를 컨테이너에 마운트해 재다운로드를 방지합니다.
HF_CACHE="${HF_HOME:-${HOME}/.cache/huggingface}"

case "${1:-}" in
  stop)
    docker stop "${CONTAINER_NAME}"
    docker rm "${CONTAINER_NAME}"
    echo "[INFO] Container ${CONTAINER_NAME} stopped and removed."
    exit 0
    ;;
  build)
    echo "[INFO] Building local image ${LOCAL_IMAGE} from Dockerfile.solar..."
    docker build -f "${SCRIPT_DIR}/Dockerfile.solar" -t "${LOCAL_IMAGE}" "${SCRIPT_DIR}"
    echo "[INFO] Build complete: ${LOCAL_IMAGE}"
    exit 0
    ;;
  start|""|run) ;;
  *)
    echo "Usage: $0 [start|stop|build]"
    exit 1
    ;;
esac

echo "[INFO] Base image : ${DOCKER_IMAGE}"
echo "[INFO] Local image: ${LOCAL_IMAGE}"
echo "[INFO] Model      : ${MODEL_PATH}"
echo "[INFO] GPUs       : device=${GPU_DEVICES}  TP=${TP_SIZE}"
echo "[INFO] Endpoint   : ${HOST}:${PORT}  served-as=${SERVED_MODEL_NAME}"
echo "[INFO] HF cache   : ${HF_CACHE}"

# 로컬 이미지가 없으면 자동 빌드 (transformers 업그레이드 포함)
if ! docker image inspect "${LOCAL_IMAGE}" &>/dev/null; then
  echo "[INFO] Local image not found. Building ${LOCAL_IMAGE} from Dockerfile.solar..."
  docker build -f "${SCRIPT_DIR}/Dockerfile.solar" -t "${LOCAL_IMAGE}" "${SCRIPT_DIR}"
  echo "[INFO] Build complete."
fi

# 동명의 잔여 컨테이너가 있으면 제거
if docker inspect "${CONTAINER_NAME}" &>/dev/null; then
  echo "[INFO] Removing existing container '${CONTAINER_NAME}'..."
  docker rm -f "${CONTAINER_NAME}"
fi

DETACH=()
[[ "${1:-}" == "start" ]] && DETACH=("-d")

docker run "${DETACH[@]}" \
  --name "${CONTAINER_NAME}" \
  --gpus all \
  -e NVIDIA_VISIBLE_DEVICES="${GPU_DEVICES}" \
  --ipc=host \
  -p "${PORT}:${PORT}" \
  -v "${HF_CACHE}:/root/.cache/huggingface" \
  "${LOCAL_IMAGE}" \
    "${MODEL_PATH}" \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser solar_open \
    --reasoning-parser solar_open \
    --logits-processors "vllm.model_executor.models.parallel_tool_call_logits_processor:ParallelToolCallLogitsProcessor" \
    --logits-processors "vllm.model_executor.models.solar_open_logits_processor:SolarOpenTemplateLogitsProcessor" \
    --host "${HOST}" \
    --port "${PORT}" \
    --dtype bfloat16 \
    --tensor-parallel-size "${TP_SIZE}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTIL}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --api-key EMPTY

if [[ "${1:-}" == "start" ]]; then
  echo "[INFO] vLLM started in background. container=${CONTAINER_NAME}"
  echo "       Logs : docker logs -f ${CONTAINER_NAME}"
fi
