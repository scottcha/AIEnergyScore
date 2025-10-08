# Option A Implementation: Environment Variable Backend Switching

**Status:** ✅ Implemented
**Date:** 2025-10-07
**Implementation:** Environment Variable-based backend selection for AIEnergyScore

## Overview

This implementation enables AIEnergyScore to switch between different benchmark backends using the `BENCHMARK_BACKEND` environment variable, providing a drop-in replacement capability without code changes.

## Supported Backends

1. **optimum** (default) - HuggingFace's optimum-benchmark framework
2. **pytorch** - ai_energy_benchmarks with PyTorch backend
3. **vllm** - ai_energy_benchmarks with vLLM backend (requires vLLM server)

## Implementation Changes

### 1. Modified Files

#### `entrypoint.sh`
- Added `BENCHMARK_BACKEND` environment variable with default value "optimum"
- Implemented conditional routing based on backend selection
- Added validation and error handling for unknown backends
- Required `VLLM_ENDPOINT` validation for vLLM backend

#### `Dockerfile`
- Added installation of `ai_energy_benchmarks[pytorch]` package
- Maintained backward compatibility with optimum-benchmark
- Added graceful fallback if ai_energy_benchmarks is not available

#### `summarize_gpu_wh.py`
- Enhanced to detect and handle both result formats automatically
- Added `detect_format()` function to identify result type
- Added `extract_ai_energy_benchmarks_format()` for new format
- Added `extract_optimum_format()` for existing format
- Maintains backward compatibility with existing tools

### 2. New Capabilities

- **Format Detection**: Automatically detects optimum-benchmark vs ai_energy_benchmarks result formats
- **Unified Output**: Both backends produce compatible `GPU_ENERGY_WH.txt` and `GPU_ENERGY_SUMMARY.json`
- **Backward Compatible**: Default behavior unchanged (uses optimum-benchmark)
- **Error Handling**: Clear error messages for invalid backend selection

## Usage Examples

### Default Behavior (optimum-benchmark)

```bash
# No changes needed - uses optimum-benchmark by default
docker run --gpus all energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

### Using PyTorch Backend

```bash
# Switch to ai_energy_benchmarks PyTorch backend
docker run --gpus all \
  -e BENCHMARK_BACKEND=pytorch \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

### Using vLLM Backend

```bash
# Requires running vLLM server
# First, start vLLM server (separate terminal):
vllm serve openai/gpt-oss-120b --port 8000

# Then run benchmark with vLLM backend:
docker run --gpus all \
  -e BENCHMARK_BACKEND=vllm \
  -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

## Result Format Compatibility

### optimum-benchmark Format
```json
{
  "preprocess": {"energy": {"gpu": 10.5, "unit": "kWh"}},
  "prefill": {"energy": {"gpu": 15.2, "unit": "kWh"}},
  "decode": {"energy": {"gpu": 99.7, "unit": "kWh"}}
}
```

### ai_energy_benchmarks Format
```json
{
  "energy": {
    "gpu_energy_wh": 125.4,
    "cpu_energy_wh": 15.2,
    "total_energy_wh": 148.7
  },
  "performance": {
    "throughput_tokens_per_second": 416.7
  }
}
```

### Unified Output (Both Backends)
Both formats are converted to the same output files:

**GPU_ENERGY_WH.txt:**
```
125.40
```

**GPU_ENERGY_SUMMARY.json:**
```json
{
  "units": "Wh",
  "total": 125.4,
  "format": "ai_energy_benchmarks",
  "preprocess_wh": 0.0,
  "prefill_wh": 0.0,
  "decode_wh": 125.4
}
```

## Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BENCHMARK_BACKEND` | No | `optimum` | Backend selection: `optimum`, `pytorch`, `vllm` |
| `VLLM_ENDPOINT` | Yes (for vLLM) | - | vLLM server endpoint (e.g., `http://localhost:8000/v1`) |

## Error Handling

### Invalid Backend
```bash
docker run --gpus all -e BENCHMARK_BACKEND=invalid energy_star ...
# Output:
# Error: Unknown BENCHMARK_BACKEND=invalid
# Valid options: optimum, pytorch, vllm
# Exit code: 1
```

### Missing vLLM Endpoint
```bash
docker run --gpus all -e BENCHMARK_BACKEND=vllm energy_star ...
# Output:
# Error: VLLM_ENDPOINT environment variable must be set for vLLM backend
# Exit code: 1
```

### ai_energy_benchmarks Not Installed
```bash
# If Docker build fails to install ai_energy_benchmarks:
# Warning: ai_energy_benchmarks not available, only optimum backend will work
# (Dockerfile continues, but pytorch/vllm backends will fail at runtime)
```

## Testing

### Test Backend Switching

```bash
# Test 1: Default (optimum)
docker run --gpus all energy_star --config-name text_generation

# Test 2: PyTorch backend
docker run --gpus all -e BENCHMARK_BACKEND=pytorch energy_star --config-name text_generation

# Test 3: vLLM backend (requires server)
vllm serve openai/gpt-oss-120b &
docker run --gpus all \
  -e BENCHMARK_BACKEND=vllm \
  -e VLLM_ENDPOINT=http://localhost:8000/v1 \
  energy_star --config-name text_generation
```

### Validate Results

```bash
# All backends should produce the same output files:
ls /results/
# - GPU_ENERGY_WH.txt
# - GPU_ENERGY_SUMMARY.json
# - benchmark_report.json

# Compare energy values (should be within ±5%)
cat /results/GPU_ENERGY_WH.txt
```

## Backward Compatibility

✅ **Fully backward compatible**
- Default behavior unchanged (uses optimum-benchmark)
- Existing configurations work without modification
- Existing scripts and CI/CD pipelines unaffected
- Result format compatible with downstream tools

## Known Limitations

1. **ai_energy_benchmarks Package**: Currently assumes package is available. The Dockerfile includes a fallback, but runtime will fail if the package is not installed and pytorch/vllm backends are selected.

2. **Phase Breakdown**: ai_energy_benchmarks doesn't separate energy by phase (preprocess/prefill/decode). All energy is assigned to the "decode" phase for compatibility.

3. **Configuration Format**: The ai-energy-benchmark CLI expects a specific config format. Current implementation passes through arguments, but may require config adaptation.

## Next Steps

1. **Publish ai_energy_benchmarks**: Package needs to be published to PyPI for easy installation
2. **CLI Compatibility**: Verify ai-energy-benchmark CLI config compatibility with existing YAML files
3. **Integration Testing**: Comprehensive testing on H100/B200 hardware
4. **Documentation**: Update main README with backend switching instructions
5. **CI/CD**: Add backend switching tests to CI pipeline

## Related Documents

- [Backend Switching Strategy](/home/scott/src/AIEnergyScore/ai_helpers/backend_switching_strategy.md) - Full design document
- [AIEnergyScore README](/home/scott/src/AIEnergyScore/README.md) - Main documentation

## Support

For issues or questions:
- Check ai_energy_benchmarks documentation
- Review design document section 4.1 (AIEnergyScore Integration)
- Open GitHub issue with backend selection details
