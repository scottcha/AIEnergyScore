# Manual Test Commands for AIEnergyScore

## Option 1: Use Default Config (Simplest)

Run with the default text_generation config from the container:

```bash
cd /mnt/storage/src/AIEnergyScore
mkdir -p validation_results

docker run --gpus all \
  --shm-size 8g \
  -v $(pwd)/validation_results:/results \
  ai_energy_score \
  --config-name text_generation \
  hydra.run.dir=/results
```

This will use whatever defaults are in the `reasoning_test` branch config.

## Option 2: Override Only the Model

```bash
docker run --gpus all \
  --shm-size 8g \
  -v $(pwd)/validation_results:/results \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-20b \
  backend.processor=openai/gpt-oss-20b \
  hydra.run.dir=/results
```

## Option 3: Check What Configs Exist

List available configs in the container:

```bash
docker run --gpus all --rm ai_energy_score ls -la /optimum-benchmark/energy_star/
```

## Option 4: Inspect the Config

See what the actual config looks like:

```bash
docker run --gpus all --rm ai_energy_score cat /optimum-benchmark/energy_star/text_generation.yaml
```

## Debugging

If you get config errors, try:

```bash
# See full config structure
docker run --gpus all --rm \
  ai_energy_score \
  --config-name text_generation \
  --cfg job

# See available overrides
docker run --gpus all --rm \
  ai_energy_score \
  --config-name text_generation \
  --help
```

## Recommended Approach

1. First, inspect the actual config: Option 4
2. Then run with minimal overrides: Option 1 or 2
3. Adjust based on what's actually in the config file
