# Docker Volume Mounting Guide

This guide explains how to properly run the AIEnergyScore Docker container with volume mounts for caching and results persistence.

## Quick Reference

### Standard Run Command

```bash
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b
```

## Why Run as Current User?

Running the container as the current user (instead of root) provides:

1. **Proper file permissions**: Files created in mounted volumes have correct ownership
2. **Security**: Avoids running processes as root unnecessarily
3. **Convenience**: No need to `sudo chown` result files after benchmarks

## Volume Mounts Explained

### HuggingFace Cache Mount

```bash
-v ~/.cache/huggingface:/home/user/.cache/huggingface
```

**Purpose**: Reuse downloaded model weights across container runs

**Benefits**:
- No re-downloading of multi-GB model files
- Faster startup time
- Reduced bandwidth usage
- Persistent cache across container recreations

**How it works**:
- Host cache: `~/.cache/huggingface` (your local HuggingFace cache)
- Container path: `/home/user/.cache/huggingface`
- `HOME=/home/user` tells HuggingFace libraries where to find the cache

### Results Directory Mount

```bash
-v $(pwd)/results:/results
```

**Purpose**: Persist benchmark outputs to host filesystem

**Benefits**:
- Results survive container deletion
- Easy access to output files from host
- Can be used for subsequent analysis

**Output files**:
- `benchmark_report.json` - Detailed benchmark results
- `GPU_ENERGY_WH.txt` - Energy consumption summary
- `GPU_ENERGY_SUMMARY.json` - JSON-formatted energy summary
- Emission tracking logs (if enabled)

## User and Permissions

### Running as Current User

```bash
--user $(id -u):$(id -g)
```

- `$(id -u)` - Your user ID (e.g., 1000)
- `$(id -g)` - Your group ID (e.g., 1000)

This ensures the container process runs with your user permissions.

### Setting HOME Directory

```bash
-e HOME=/home/user
```

**Why needed**:
- HuggingFace libraries check `$HOME/.cache/huggingface` for cached models
- Without this, the container wouldn't know where to find the cache
- Must match the path used in the cache volume mount

## Advanced Options

### Read-Only Cache Mount

If you want to prevent the container from downloading new models (only use existing cache):

```bash
-v ~/.cache/huggingface:/home/user/.cache/huggingface:ro
```

The `:ro` flag makes the mount read-only.

### Using Host Home Directory

Alternative approach that mounts your entire home directory:

```bash
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v $HOME:$HOME \
  -v $(pwd)/results:/results \
  -e HOME=$HOME \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b
```

**Pros**: All paths match exactly between host and container
**Cons**: Less isolated, exposes entire home directory to container

### Custom Cache Location

If your HuggingFace cache is in a non-standard location:

```bash
# Set custom cache location
export HF_HOME=/path/to/custom/cache

docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v $HF_HOME:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b
```

## Troubleshooting

### Permission Denied Errors

If you see permission errors:

```bash
# Ensure results directory exists and is writable
mkdir -p ./results
chmod 755 ./results

# Verify cache directory permissions
ls -la ~/.cache/huggingface
```

### Models Still Downloading

If models re-download despite cache mounting:

1. Verify the cache mount path: `docker run ... ls -la /home/user/.cache/huggingface`
2. Check that `HOME` environment variable is set: `docker run ... env | grep HOME`
3. Ensure models exist in host cache: `ls -la ~/.cache/huggingface/hub/`

### Cannot Access Results

If results aren't appearing in `./results`:

1. Check if directory was created: `ls -la ./results`
2. Verify mount succeeded: `docker inspect <container_id> | grep Mounts -A 10`
3. Check container logs: `docker logs <container_id>`

## Backend-Specific Examples

### PyTorch Backend

```bash
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  -e BENCHMARK_BACKEND=pytorch \
  ai_energy_score \
  --config-name text_generation \
  scenario.num_samples=100 \
  backend.model=openai/gpt-oss-20b
```

### vLLM Backend

```bash
# vLLM backend doesn't need HF cache (model runs on external server)
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  -e BENCHMARK_BACKEND=vllm \
  -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

Note: vLLM backend only needs results mount, not HF cache, since models are loaded by the external vLLM server.

## Best Practices

1. **Always mount results directory** - Prevents loss of benchmark data
2. **Mount HF cache for PyTorch/optimum backends** - Saves time and bandwidth
3. **Run as current user** - Maintains proper file ownership
4. **Pre-create results directory** - Avoids permission issues
5. **Use absolute paths** - More reliable than relative paths in volume mounts

## Summary

The recommended docker run pattern:

```bash
# Create results directory
mkdir -p ./results

# Run with all proper mounts
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  ai_energy_score \
  --config-name <config> \
  backend.model=<model>
```

This ensures:
- ✅ Models are cached and reused
- ✅ Results are persisted to host
- ✅ Files have correct ownership
- ✅ No root privileges needed
