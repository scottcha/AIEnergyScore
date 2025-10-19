# AI Helpers Directory

This directory contains support scripts and documentation for AIEnergyScore development and deployment.

## Docker Usage

### `run_docker.sh`
**Location**: `AIEnergyScore/run_docker.sh` (parent directory)

**Purpose**: Simplified wrapper for running AIEnergyScore Docker container with proper volume mounts

**Usage**:
```bash
cd AIEnergyScore
./run_docker.sh [OPTIONS] --config-name text_generation backend.model=openai/gpt-oss-20b
```

**Options**:
- `-n, --num-samples NUM` - Number of prompts to test (default: 20)
- `-h, --help` - Show help message

**Features**:
- Automatically runs container as current user (not root)
- Mounts HuggingFace cache from `~/.cache/huggingface` to avoid re-downloading models
- Creates and mounts results directory
- Supports all benchmark backends (optimum, pytorch, vllm)
- Configurable number of test prompts with sensible default

**Environment Variables**:
- `DOCKER_IMAGE` - Override default image name (default: `energy_star`)
- `RESULTS_DIR` - Custom results directory (default: `./results`)
- `HF_HOME` - Custom HuggingFace cache location (default: `~/.cache/huggingface`)
- `BENCHMARK_BACKEND` - Backend selection: `optimum`, `pytorch`, `vllm` (default: `optimum`)

**Examples**:
```bash
# Default 20 samples
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b

# Test with 100 samples
./run_docker.sh -n 100 --config-name text_generation backend.model=openai/gpt-oss-120b

# PyTorch backend with 50 samples
BENCHMARK_BACKEND=pytorch ./run_docker.sh -n 50 --config-name text_generation backend.model=openai/gpt-oss-20b

# Custom results location
RESULTS_DIR=/tmp/my_results ./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-120b

# vLLM backend (no HF cache needed)
BENCHMARK_BACKEND=vllm VLLM_ENDPOINT=http://localhost:8000/v1 ./run_docker.sh --config-name text_generation
```

### `DOCKER_VOLUME_MOUNTING.md`
**Purpose**: Comprehensive guide on Docker volume mounting strategy

**Contents**:
- Why run as current user instead of root
- Volume mount explanations (HF cache, results directory)
- Troubleshooting permission issues
- Advanced mounting options
- Backend-specific examples

## Integration and Validation Testing

### Python Unit Tests (MOVED TO `tests/`)
Python test files have been migrated to `../tests/` for integration with pytest regression suite:
- `test_batch_runner.py` → `tests/test_batch_runner.py` - Batch runner component tests
- `test_reasoning_parameters.py` → `tests/test_reasoning_parameters.py` - Reasoning parameter handling tests

**Run regression tests:**
```bash
cd AIEnergyScore
.venv/bin/pytest tests/ -v -m "not integration"
```

### Shell Integration Tests (Kept in `ai_helpers/`)

#### `test_reasoning_energy_integration.sh`
**Purpose**: Full integration test verifying different reasoning levels produce different energy results

**Usage**:
```bash
cd AIEnergyScore
./ai_helpers/test_reasoning_energy_integration.sh [num_prompts]

# Example: Test with 5 prompts
./ai_helpers/test_reasoning_energy_integration.sh 5

# Example: Test with 20 prompts for stable measurements
./ai_helpers/test_reasoning_energy_integration.sh 20
```

**What it does**:
- Runs gpt-oss-20b at all reasoning levels (High, Low, Off)
- Analyzes token counts and energy consumption
- Verifies different reasoning levels produce measurably different results
- Provides detailed pass/fail analysis

**When to use**: After making changes to reasoning parameter handling or batch runner logic

#### `test_batch_runner_docker.sh`
**Purpose**: Quick smoke test for batch_runner.py with Docker backend

**Usage**:
```bash
cd AIEnergyScore
./ai_helpers/test_batch_runner_docker.sh
```

**What it does**:
- Tests batch_runner.py with pytorch (docker) backend
- Runs a single prompt for quick validation
- Verifies output files are created correctly

**When to use**: After changes to Docker integration or batch_runner.py

#### `run_validation_test.sh`
**Purpose**: Legacy validation script for PyTorch backend compatibility testing

**Usage**:
```bash
./ai_helpers/run_validation_test.sh
```

**Note**: Consider using the pytest tests or integration scripts above instead

## Documentation

### `MANUAL_RUN.md`
**Purpose**: Manual instructions for running benchmarks without helper scripts

Useful for understanding the underlying Docker commands and troubleshooting.

### `OPTION_A_IMPLEMENTATION.md`
**Purpose**: Implementation details for backend selection architecture

Technical documentation for developers working on backend integration.

### `backend_switching_strategy.md`
**Purpose**: Architecture documentation for multi-backend support

Explains how AIEnergyScore supports multiple benchmark backends (optimum-benchmark, ai_energy_benchmarks with PyTorch, ai_energy_benchmarks with vLLM).

## Quick Reference

### Running a Benchmark

**Easiest way** (recommended):
```bash
cd AIEnergyScore
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

**With custom number of samples**:
```bash
cd AIEnergyScore
./run_docker.sh -n 100 --config-name text_generation backend.model=openai/gpt-oss-20b
```

**Manual way** (for advanced users):
```bash
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b \
  scenario.num_samples=20
```

### Key Concepts

1. **Volume Mounting**: Reuse downloaded models and persist results
2. **User Permissions**: Run as current user for proper file ownership
3. **Backend Selection**: Choose between optimum-benchmark, PyTorch, or vLLM
4. **HuggingFace Cache**: Avoid re-downloading multi-GB model weights

## File Organization

```
AIEnergyScore/
├── run_docker.sh                              # Docker wrapper script (MAIN ENTRY POINT)
├── tests/                                     # Pytest regression test suite
│   ├── test_batch_runner.py                  # Batch runner component tests
│   ├── test_reasoning_parameters.py          # Reasoning parameter tests
│   └── test_e2e_small_models.py              # End-to-end tests
└── ai_helpers/                                # Support scripts and documentation
    ├── README.md                              # This file
    ├── test_reasoning_energy_integration.sh   # Integration test script
    ├── test_batch_runner_docker.sh            # Docker backend smoke test
    ├── run_validation_test.sh                 # Legacy validation script
    ├── DOCKER_VOLUME_MOUNTING.md              # Volume mounting guide
    ├── MANUAL_RUN.md                          # Manual run instructions
    ├── OPTION_A_IMPLEMENTATION.md             # Backend implementation docs
    ├── backend_switching_strategy.md          # Multi-backend architecture
    └── README_REASONING_TESTS.md              # Reasoning test documentation
```

## Contributing

When adding new helper scripts or documentation:

1. Place them in this `ai_helpers/` directory
2. Update this README with a description
3. Ensure scripts are executable: `chmod +x script_name.sh`
4. Follow existing naming conventions
5. Add usage examples
