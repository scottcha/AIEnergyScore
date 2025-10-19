# run_docker.sh Update Summary

## Changes Made

### 1. **Moved Script Location**
- **From**: `AIEnergyScore/ai_helpers/run_docker.sh`
- **To**: `AIEnergyScore/run_docker.sh` (parent directory)
- **Reason**: More accessible location for primary entry point

### 2. **Added Number of Samples Parameter**
- New option: `-n, --num-samples NUM`
- Default: **20 prompts**
- Overrides `scenario.num_samples` parameter automatically

### 3. **Enhanced Environment Variable Handling**
- Now passes `BENCHMARK_BACKEND` to Docker container
- Passes `VLLM_ENDPOINT` when using vLLM backend
- Maintains all existing environment variable support

### 4. **Added Help Message**
- `-h, --help` flag shows usage instructions
- Displays all options and environment variables
- Includes examples

## New Usage

### Basic Usage with Defaults (20 samples)
```bash
cd AIEnergyScore
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

### Custom Number of Samples
```bash
# Test with 5 samples (quick test)
./run_docker.sh -n 5 --config-name text_generation backend.model=openai/gpt-oss-20b

# Full benchmark with 100 samples
./run_docker.sh --num-samples 100 --config-name text_generation backend.model=openai/gpt-oss-120b
```

### With Different Backends
```bash
# PyTorch backend with 50 samples
BENCHMARK_BACKEND=pytorch ./run_docker.sh -n 50 --config-name text_generation backend.model=openai/gpt-oss-20b

# vLLM backend (uses external server)
BENCHMARK_BACKEND=vllm VLLM_ENDPOINT=http://localhost:8000/v1 ./run_docker.sh -n 20 --config-name text_generation
```

## Script Features

### Automatic Configuration
1. **Volume Mounts**: Automatically mounts HF cache and results directory
2. **User Permissions**: Runs as current user (not root)
3. **Directory Creation**: Creates results directory if it doesn't exist
4. **Backend Detection**: Conditionally mounts HF cache based on backend type

### Output Display
Shows configuration before running:
```
============================================
AIEnergyScore Docker Runner
============================================
Image:         energy_star
Backend:       pytorch
Num Samples:   20
Results dir:   /mnt/storage/src/AIEnergyScore/results
HF Cache:      /home/scott/.cache/huggingface
User:          1000:1000
============================================
```

## Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `-n, --num-samples NUM` | Number of prompts to test | 20 |
| `-h, --help` | Show help message | - |

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `DOCKER_IMAGE` | Docker image name | `energy_star` |
| `RESULTS_DIR` | Results output directory | `./results` |
| `HF_HOME` | HuggingFace cache location | `~/.cache/huggingface` |
| `BENCHMARK_BACKEND` | Backend: optimum, pytorch, vllm | `optimum` |
| `VLLM_ENDPOINT` | vLLM server endpoint (vLLM only) | - |

## Testing Results

### Test 1: PyTorch Backend with 2 Samples
```bash
BENCHMARK_BACKEND=pytorch ./run_docker.sh -n 2 --config-name text_generation backend.model=openai/gpt-oss-20b
```

**Results**:
- ✅ Correctly used PyTorch backend
- ✅ Processed exactly 2 prompts
- ✅ Generated 0.77 Wh energy consumption
- ✅ All files owned by current user
- ✅ Results saved to `./results/`

### Test 2: Default Behavior
```bash
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

**Expected**:
- Backend: optimum (default)
- Samples: 20 (default)
- HF cache mounted and reused

## Migration from Old Script

### Before (ai_helpers location)
```bash
cd AIEnergyScore/ai_helpers
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

### After (parent directory)
```bash
cd AIEnergyScore
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

## Documentation Updates

Updated files:
1. ✅ `README.md` - Updated script location and added num-samples examples
2. ✅ `ai_helpers/README.md` - Updated reference to script location
3. ✅ `ai_helpers/DOCKER_VOLUME_MOUNTING.md` - Updated references
4. ✅ `ai_helpers/VOLUME_MOUNTING_CHANGES.md` - Updated script location

## Key Improvements

1. **Simpler Usage**: Script in parent directory is more accessible
2. **Sensible Defaults**: 20 samples is a good balance for testing
3. **Flexibility**: Easy to customize sample count for quick tests or full benchmarks
4. **Better UX**: Help message and clear configuration display
5. **Robust**: Proper environment variable forwarding to container

## Common Use Cases

### Quick Development Test (5 samples)
```bash
./run_docker.sh -n 5 --config-name text_generation backend.model=openai/gpt-oss-20b
```
**Time**: ~30 seconds

### Standard Test (20 samples - default)
```bash
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```
**Time**: ~2 minutes

### Full Benchmark (100 samples)
```bash
./run_docker.sh -n 100 --config-name text_generation backend.model=openai/gpt-oss-120b
```
**Time**: ~10 minutes

### Production Benchmark (1000 samples)
```bash
./run_docker.sh -n 1000 --config-name text_generation backend.model=openai/gpt-oss-120b
```
**Time**: ~1-2 hours

## Troubleshooting

### Permission Denied on /results/emissions

**Problem**: Results directory owned by root from previous runs

**Solution**: Remove and recreate results directory
```bash
rm -rf results && mkdir -p results
./run_docker.sh -n 5 --config-name text_generation backend.model=openai/gpt-oss-20b
```

### Models Still Downloading

**Problem**: HF cache not being reused

**Solution**: Verify cache mount in script output
```
HF Cache:      /home/scott/.cache/huggingface
```

Check that models exist:
```bash
ls -lh ~/.cache/huggingface/hub/
```

## Summary

The updated `run_docker.sh` script provides a streamlined, user-friendly way to run AIEnergyScore benchmarks with:
- Sensible defaults (20 samples)
- Easy customization (`-n` flag)
- Proper permissions (runs as current user)
- HuggingFace cache reuse (saves time and bandwidth)
- Clear configuration display
- Help documentation (`--help`)
