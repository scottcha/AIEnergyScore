# Volume Mounting Implementation Summary

## Overview

Updated AIEnergyScore Docker documentation and tooling to run containers as current user (not root) with proper HuggingFace cache mounting to avoid re-downloading model weights.

## Changes Made

### 1. Updated Documentation (`README.md`)

**Location**: `/mnt/storage/src/AIEnergyScore/README.md`

**Changes**:
- Added "Quick Start with Helper Script" section pointing to `run_docker.sh`
- Updated all docker run examples to include:
  - `--user $(id -u):$(id -g)` - Run as current user
  - `-v ~/.cache/huggingface:/home/user/.cache/huggingface` - Mount HF cache
  - `-v $(pwd)/results:/results` - Mount results directory
  - `-e HOME=/home/user` - Set HOME for HF cache detection
- Updated examples for all backends: optimum (default), pytorch, vllm

### 2. Created Helper Script (`run_docker.sh`)

**Location**: `/mnt/storage/src/AIEnergyScore/run_docker.sh`

**Features**:
- Automatically handles all volume mounts
- Runs container as current user
- Creates results directory if needed
- Configurable number of test prompts (default: 20)
- Supports environment variable configuration:
  - `DOCKER_IMAGE` - Override image name
  - `RESULTS_DIR` - Custom results location
  - `HF_HOME` - Custom HF cache path
  - `BENCHMARK_BACKEND` - Backend selection
- Conditionally mounts HF cache (not needed for vLLM backend)

**Usage**:
```bash
cd AIEnergyScore
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
./run_docker.sh -n 100 --config-name text_generation backend.model=openai/gpt-oss-120b
```

### 3. Created Documentation Guide (`DOCKER_VOLUME_MOUNTING.md`)

**Location**: `/mnt/storage/src/AIEnergyScore/ai_helpers/DOCKER_VOLUME_MOUNTING.md`

**Contents**:
- Quick reference examples
- Explanation of why to run as current user
- Detailed volume mount explanations
- Troubleshooting guide
- Advanced options (read-only mounts, custom paths)
- Backend-specific examples
- Best practices

### 4. Created ai_helpers README (`ai_helpers/README.md`)

**Location**: `/mnt/storage/src/AIEnergyScore/ai_helpers/README.md`

**Contents**:
- Documentation of all helper scripts and docs
- Quick reference for common tasks
- File organization overview
- Contributing guidelines

## Benefits

### For Users

1. **No Re-downloading Models**
   - Models cached in `~/.cache/huggingface` are reused
   - Saves bandwidth and time on subsequent runs
   - Typical model sizes: 10-100GB

2. **Proper File Permissions**
   - Files created by container owned by current user
   - No need to `sudo chown` after benchmarks
   - Direct access to results without permission issues

3. **Security**
   - Container doesn't run as root
   - Follows Docker best practices
   - Minimal privilege principle

4. **Convenience**
   - Helper script handles all configuration
   - Environment variables for customization
   - Results persisted to host filesystem

### For Development

1. **Consistent Usage Pattern**
   - All examples show proper volume mounting
   - Helper script provides single source of truth
   - Easy to update if patterns change

2. **Better Documentation**
   - Comprehensive troubleshooting guide
   - Advanced usage examples
   - Clear explanation of design decisions

## Migration Guide

### Old Pattern (Root User, No Cache)

```bash
# DON'T USE - Old pattern without cache mounting
docker run --gpus all --shm-size 1g ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b
```

**Problems**:
- Runs as root
- Re-downloads models every time
- Results owned by root
- No persistence

### New Pattern (Current User, With Cache)

```bash
# RECOMMENDED - New pattern with cache mounting
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b
```

**Or use helper script**:
```bash
cd AIEnergyScore/ai_helpers
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

## Testing Verification

To verify the changes work correctly:

1. **Build the image**:
   ```bash
   cd /mnt/storage/src
   docker build -f AIEnergyScore/Dockerfile -t ai_energy_score .
   ```

2. **Run with helper script**:
   ```bash
   cd AIEnergyScore/ai_helpers
   ./run_docker.sh --config-name text_generation scenario.num_samples=10 backend.model=openai/gpt-oss-20b
   ```

3. **Verify cache mount**:
   ```bash
   # Check if models are being cached
   ls -la ~/.cache/huggingface/hub/

   # Check results ownership
   ls -la ./results/
   ```

4. **Second run should be faster**:
   - First run: Downloads model (slow)
   - Second run: Uses cached model (fast)

## Technical Details

### Volume Mount Paths

- **Host HF cache**: `~/.cache/huggingface` (user's local cache)
- **Container HF cache**: `/home/user/.cache/huggingface`
- **Host results**: `$(pwd)/results` (current directory)
- **Container results**: `/results` (hardcoded in entrypoint.sh)

### User Mapping

- `$(id -u)` - Current user ID (e.g., 1000)
- `$(id -g)` - Current group ID (e.g., 1000)
- Container process runs with these IDs
- Files created in mounted volumes have correct ownership

### HuggingFace Cache Detection

HuggingFace libraries search for cache in this order:
1. `HF_HOME` environment variable
2. `$HOME/.cache/huggingface`
3. Default system cache

By setting `HOME=/home/user` and mounting cache at `/home/user/.cache/huggingface`, we ensure HF finds the cache.

## Files Modified

1. `/mnt/storage/src/AIEnergyScore/README.md` - Updated main documentation
2. Created `/mnt/storage/src/AIEnergyScore/ai_helpers/run_docker.sh` - Helper script
3. Created `/mnt/storage/src/AIEnergyScore/ai_helpers/DOCKER_VOLUME_MOUNTING.md` - Detailed guide
4. Created `/mnt/storage/src/AIEnergyScore/ai_helpers/README.md` - ai_helpers index

## Next Steps

Users can now:
1. Use the helper script for simplified execution
2. Reference the comprehensive mounting guide for advanced usage
3. Run containers without root privileges
4. Reuse cached models across runs
5. Access results without permission issues
