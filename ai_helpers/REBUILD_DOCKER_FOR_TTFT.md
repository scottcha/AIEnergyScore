# Rebuild Docker Image with TTFT Tracking

## Why Rebuild?

The `batch_runner.py` runs benchmarks inside Docker. The current Docker image has the **old version** of `ai_energy_benchmarks` without TTFT tracking. To see TTFT metrics in your CSV results, you need to rebuild the Docker image with the updated package.

## Quick Verification

To verify the local package has TTFT tracking:

```bash
cd /home/scott/src/AIEnergyScore
source .venv/bin/activate
python ai_helpers/test_ttft_working.py
```

You should see: ✅ SUCCESS: 'enable_streaming' parameter found!

## Steps to Rebuild Docker Image

### 1. Ensure Updated Wheel Exists

```bash
cd /home/scott/src/ai_energy_benchmarks
ls -lh dist/ai_energy_benchmarks-*.whl
```

You should see a wheel file (created earlier).

### 2. Rebuild Docker Image

The Dockerfile has been updated to use the local wheel (Option B).

```bash
cd /home/scott/src/AIEnergyScore

# Build the Docker image
# Note: The build context needs to include both AIEnergyScore and ai_energy_benchmarks
cd /home/scott/src

docker build -f AIEnergyScore/Dockerfile \
  -t aienergyscore:ttft-enabled \
  .
```

### 3. Update run_docker.sh (if needed)

Check if `run_docker.sh` needs to reference the new image name:

```bash
cd /home/scott/src/AIEnergyScore
grep "docker run" run_docker.sh
```

If it uses a specific image name, update it to use `aienergyscore:ttft-enabled`.

### 4. Run a Test Benchmark

```bash
cd /home/scott/src/AIEnergyScore
source .venv/bin/activate

python batch_runner.py \
    --model-name "Hunyuan-1.8B-Instruct" \
    --output-dir ./results/ttft_test \
    --num-prompts 5
```

### 5. Verify TTFT in Results

Check the individual run CSV:

```bash
head -2 results/ttft_test/individual_runs/*/benchmark_results.csv
```

Look for the `summary_avg_time_to_first_token` column - it should have **non-zero values** now!

Check the master results:

```bash
head -2 results/ttft_test/master_results.csv
```

Look for the `avg_time_to_first_token` column.

## Alternative: Quick Test Without Docker

If you want to test TTFT tracking immediately without rebuilding Docker:

### Option 1: Use Direct Execution (Not Docker)

This requires modifying `batch_runner.py` to skip Docker execution. **Only for testing - not recommended for production.**

### Option 2: Use ai_energy_benchmarks Directly

```bash
cd /home/scott/src/ai_energy_benchmarks
source .venv/bin/activate

# Run a quick benchmark
python -c "
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
backend = PyTorchBackend(model='tencent/Hunyuan-1.8B-Instruct', device='cuda')
result = backend.run_inference('Hello, how are you?', max_tokens=50, enable_streaming=True)
print(f'TTFT: {result[\"time_to_first_token\"]:.4f}s')
print(f'Total latency: {result[\"latency_seconds\"]:.4f}s')
"
```

## Expected Results

After rebuilding Docker and running benchmarks, you should see:

### In benchmark_results.csv:
- New column: `summary_avg_time_to_first_token` with values like `0.1500`, `0.2200`, etc.

### In master_results.csv:
- New column: `avg_time_to_first_token` with non-zero values
- Renamed column: `avg_total_time` (from `avg_latency_seconds`)

## Typical TTFT Values

- **Small models (< 10B)**: 50-200ms
- **Medium models (10-70B)**: 100-500ms
- **Large models (> 70B)**: 200-2000ms

If you see `0.0000`, the Docker image likely still has the old package version.

## Troubleshooting

### Problem: Still seeing 0.0000 for TTFT

**Solution**: Verify Docker is using the new image:
```bash
docker images | grep aienergyscore
```

Make sure `run_docker.sh` references the correct image name.

### Problem: Docker build fails

**Solution**: Ensure the build context includes both directories:
```bash
cd /home/scott/src  # Parent of both AIEnergyScore and ai_energy_benchmarks
docker build -f AIEnergyScore/Dockerfile -t aienergyscore:ttft-enabled .
```

### Problem: Wheel file not found during build

**Solution**: Verify the wheel exists:
```bash
ls -lh /home/scott/src/ai_energy_benchmarks/dist/ai_energy_benchmarks-*.whl
```

If missing, rebuild it:
```bash
cd /home/scott/src/ai_energy_benchmarks
./build_wheel.sh
```

## Summary

1. ✅ TTFT tracking implemented in `ai_energy_benchmarks` package
2. ✅ Local package updated with TTFT tracking
3. ✅ `ResultsAggregator` updated with new columns
4. ✅ Dockerfile updated to use local wheel
5. ⏳ **TODO**: Rebuild Docker image
6. ⏳ **TODO**: Run test benchmark to verify TTFT values

Once Docker is rebuilt, all future benchmark runs will automatically include TTFT metrics!
