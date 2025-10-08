#!/bin/bash
# Build AIEnergyScore Docker image with ai_energy_benchmarks support
#
# This script builds from the parent directory to allow access to both
# AIEnergyScore and ai_energy_benchmarks directories

set -e

echo "Building AIEnergyScore Docker image with ai_energy_benchmarks support..."
echo "Build context: /home/scott/src/"
echo ""

cd /home/scott/src/

docker build \
    -f AIEnergyScore/Dockerfile \
    -t energy_star \
    .

echo ""
echo "✓ Docker image 'energy_star' built successfully"
echo ""
echo "Usage examples:"
echo "  # Default (optimum-benchmark):"
echo "  docker run --gpus all energy_star --config-name text_generation"
echo ""
echo "  # PyTorch backend (ai_energy_benchmarks):"
echo "  docker run --gpus all -e BENCHMARK_BACKEND=pytorch energy_star --config-name text_generation"
echo ""
echo "  # vLLM backend (ai_energy_benchmarks):"
echo "  docker run --gpus all -e BENCHMARK_BACKEND=vllm -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 energy_star --config-name text_generation"
