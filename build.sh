#!/bin/bash
# Build AIEnergyScore Docker image with ai_energy_benchmarks support
#
# This script builds the Docker image, installing ai_energy_benchmarks from TestPyPI

set -e

echo "Building AIEnergyScore Docker image with ai_energy_benchmarks from TestPyPI..."
echo "Build context: ~/src/"
echo ""

cd ~/src/

# Default version (can be overridden)
AI_ENERGY_BENCHMARKS_VERSION=${AI_ENERGY_BENCHMARKS_VERSION:-0.0.1rc1}

echo "Using ai_energy_benchmarks version: ${AI_ENERGY_BENCHMARKS_VERSION}"
echo ""

# Build Docker image
docker build \
    -f AIEnergyScore/Dockerfile \
    --build-arg AI_ENERGY_BENCHMARKS_VERSION=${AI_ENERGY_BENCHMARKS_VERSION} \
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
