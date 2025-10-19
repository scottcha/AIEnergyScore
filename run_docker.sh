#!/bin/bash
# Helper script to run AIEnergyScore Docker container with proper volume mounts
# Usage: ./run_docker.sh [OPTIONS] [docker run arguments...]
#
# Options:
#   -n, --num-samples NUM    Number of prompts to test (default: 20)
#                           Note: Only applies to pytorch/vllm backends.
#                           For optimum backend, override via command line:
#                           scenario.num_samples=NUM
#   -h, --help              Show this help message
#
# Environment Variables:
#   DOCKER_IMAGE            Docker image name (default: energy_star)
#   RESULTS_DIR             Results directory (default: ./results)
#   HF_HOME                 HuggingFace cache location (default: ~/.cache/huggingface)
#   BENCHMARK_BACKEND       Backend selection: optimum, pytorch, vllm (default: optimum)
#
# Examples:
#   ./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
#   ./run_docker.sh -n 100 --config-name text_generation backend.model=openai/gpt-oss-120b
#   BENCHMARK_BACKEND=pytorch ./run_docker.sh -n 50 --config-name text_generation backend.model=openai/gpt-oss-20b
#   ./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b scenario.num_samples=100

set -e

# Default configuration
IMAGE_NAME="${DOCKER_IMAGE:-energy_star}"
RESULTS_DIR="${RESULTS_DIR:-$(pwd)/results}"
HF_CACHE="${HF_HOME:-$HOME/.cache/huggingface}"
NUM_SAMPLES=20

# Function to show help
show_help() {
    head -n 15 "$0" | grep "^#" | sed 's/^# \?//'
    exit 0
}

# Parse arguments
DOCKER_ARGS=()
while [[ $# -gt 0 ]]; do
    case $1 in
        -h|--help)
            show_help
            ;;
        -n|--num-samples)
            NUM_SAMPLES="$2"
            shift 2
            ;;
        *)
            DOCKER_ARGS+=("$1")
            shift
            ;;
    esac
done

# Create results directory if it doesn't exist
mkdir -p "${RESULTS_DIR}"

# Verify HF cache directory exists
if [ ! -d "${HF_CACHE}" ]; then
    echo "Warning: HuggingFace cache directory not found at ${HF_CACHE}"
    echo "Models will be downloaded on first run and cached at this location"
    mkdir -p "${HF_CACHE}"
fi

# Check if running with vLLM backend (doesn't need HF cache mount)
BACKEND="${BENCHMARK_BACKEND:-optimum}"
VOLUME_MOUNTS="-v ${RESULTS_DIR}:/results"

if [ "$BACKEND" != "vllm" ]; then
    # Mount HF cache for pytorch and optimum backends
    VOLUME_MOUNTS="${VOLUME_MOUNTS} -v ${HF_CACHE}:/home/user/.cache/huggingface"
fi

# Display configuration
echo "============================================"
echo "AIEnergyScore Docker Runner"
echo "============================================"
echo "Image:         ${IMAGE_NAME}"
echo "Backend:       ${BACKEND}"
if [ "$BACKEND" = "pytorch" ] || [ "$BACKEND" = "vllm" ]; then
    echo "Num Samples:   ${NUM_SAMPLES}"
else
    echo "Num Samples:   (using config default - override with scenario.num_samples=NUM)"
fi
echo "Results dir:   ${RESULTS_DIR}"
if [ "$BACKEND" != "vllm" ]; then
    echo "HF Cache:      ${HF_CACHE}"
fi
echo "User:          $(id -u):$(id -g)"
echo "============================================"
echo ""

# Build docker run command
ENV_VARS="-e HOME=/home/user -e BENCHMARK_BACKEND=${BACKEND}"
if [ "$BACKEND" = "vllm" ] && [ -n "$VLLM_ENDPOINT" ]; then
    ENV_VARS="$ENV_VARS -e VLLM_ENDPOINT=${VLLM_ENDPOINT}"
fi

# Build additional arguments
# For pytorch/vllm backends (ai_energy_benchmarks), append scenario.num_samples
# For optimum backend, let user override via command line if needed
EXTRA_ARGS=()
if [ "$BACKEND" = "pytorch" ] || [ "$BACKEND" = "vllm" ]; then
    EXTRA_ARGS+=("scenario.num_samples=${NUM_SAMPLES}")
fi

# shellcheck disable=SC2086
exec docker run --gpus all --shm-size 1g \
    --user "$(id -u):$(id -g)" \
    ${VOLUME_MOUNTS} \
    ${ENV_VARS} \
    "${IMAGE_NAME}" \
    "${DOCKER_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
