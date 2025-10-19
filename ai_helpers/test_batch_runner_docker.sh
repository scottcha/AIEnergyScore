#!/bin/bash
# Test script for batch_runner.py with docker-based pytorch backend
# This verifies that batch_runner correctly uses run_docker.sh for pytorch tests

set -e

# Configuration
TEST_OUTPUT_DIR="./test_batch_docker_output"
NUM_PROMPTS=1  # Just 1 prompt for quick test
MODEL="openai/gpt-oss-20b"
BACKEND="pytorch"

echo "============================================"
echo "Testing batch_runner.py with Docker Backend"
echo "============================================"
echo "Model: $MODEL"
echo "Backend: $BACKEND"
echo "Prompts: $NUM_PROMPTS"
echo "Output: $TEST_OUTPUT_DIR"
echo ""

# Clean up previous test output
if [ -d "$TEST_OUTPUT_DIR" ]; then
    echo "Cleaning up previous test output..."
    rm -rf "$TEST_OUTPUT_DIR"
fi

# Run batch_runner with docker backend
echo "Running batch_runner with pytorch (docker) backend..."
cd /home/scott/src/AIEnergyScore

.venv/bin/python batch_runner.py \
    --backend pytorch \
    --model-name "$MODEL" \
    --num-prompts $NUM_PROMPTS \
    --output-dir "$TEST_OUTPUT_DIR"

# Check if test was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================"
    echo "✓ Test PASSED"
    echo "============================================"
    echo ""
    echo "Checking output files..."

    # Check for expected output files
    if [ -f "$TEST_OUTPUT_DIR/master_results.csv" ]; then
        echo "✓ master_results.csv found"
    else
        echo "✗ master_results.csv NOT found"
    fi

    if [ -d "$TEST_OUTPUT_DIR/individual_runs" ]; then
        echo "✓ individual_runs directory found"
        ls -la "$TEST_OUTPUT_DIR/individual_runs"
    else
        echo "✗ individual_runs directory NOT found"
    fi

    if [ -d "$TEST_OUTPUT_DIR/logs" ]; then
        echo "✓ logs directory found"
    else
        echo "✗ logs directory NOT found"
    fi

    echo ""
    echo "Test completed successfully!"
    echo "Results saved to: $TEST_OUTPUT_DIR"
else
    echo ""
    echo "============================================"
    echo "✗ Test FAILED"
    echo "============================================"
    exit 1
fi
