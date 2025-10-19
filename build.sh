#!/bin/bash
# Build AIEnergyScore Docker image with ai_energy_benchmarks support
#
# This script builds from the parent directory to allow access to both
# AIEnergyScore and ai_energy_benchmarks directories

set -e

echo "Building AIEnergyScore Docker image with ai_energy_benchmarks support..."
echo "Build context: ~/src/"
echo ""

cd ~/src/

docker build \
    -f AIEnergyScore/Dockerfile \
    -t ai_energy_score \
    .

echo ""
echo "✓ Docker image 'ai_energy_score' built successfully"
echo ""
echo "Usage examples:"
echo "  # Default (optimum-benchmark):"
echo "  docker run --gpus all ai_energy_score --config-name text_generation"
echo ""
echo "  # PyTorch backend (ai_energy_benchmarks):"
echo "  docker run --gpus all -e BENCHMARK_BACKEND=pytorch ai_energy_score --config-name text_generation"
echo ""
echo "  # vLLM backend (ai_energy_benchmarks):"
echo "  docker run --gpus all -e BENCHMARK_BACKEND=vllm -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 ai_energy_score --config-name text_generation"
