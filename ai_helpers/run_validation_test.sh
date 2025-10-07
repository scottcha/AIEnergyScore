#!/bin/bash
# Run AIEnergyScore validation test using optimum-benchmark
# This matches the text_generation.yaml configuration

set -e

echo "========================================="
echo "AIEnergyScore Validation Test"
echo "Using optimum-benchmark PyTorch backend"
echo "========================================="
echo ""

# Configuration
RESULTS_DIR="./validation_results"

# Create results directory
mkdir -p "${RESULTS_DIR}"

echo "Configuration: text_generation (with overrides)"
echo "Model: openai/gpt-oss-20b"
echo "Dataset: EnergyStarAI/text_generation"
echo "Reasoning: Disabled"
echo "Results will be saved to: ${RESULTS_DIR}"
echo ""
echo "Running benchmark..."
echo ""

# Run via Docker using custom config (mounted into container)
docker run --gpus all \
  --shm-size 8g \
  -v $(pwd)/pytorch_validation.yaml:/optimum-benchmark/energy_star/pytorch_validation.yaml:ro \
  -v $(pwd)/${RESULTS_DIR}:/results \
  energy_star \
  --config-name pytorch_validation \
  hydra.run.dir=/results

echo ""
echo "========================================="
echo "Benchmark Complete"
echo "========================================="
echo ""
echo "Results saved to: ${RESULTS_DIR}"
echo "Key files:"
echo "  - benchmark_report.json: Full benchmark results"
echo "  - *.csv: CodeCarbon emissions data"
echo ""
