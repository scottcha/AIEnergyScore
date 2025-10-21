![AI Energy Score](/AIEnergyScore_LightBG.png)

Welcome to AI Energy Score! This is an initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.

## Key Links
- [Leaderboard](https://huggingface.co/spaces/AIEnergyScore/Leaderboard)
- [FAQ](https://huggingface.github.io/AIEnergyScore/#faq)
- [Documentation](https://huggingface.github.io/AIEnergyScore/#documentation)
- [Label Generator](https://huggingface.co/spaces/AIEnergyScore/Label)

## 🔐 Gated Model Support
AIEnergyScore now supports **automatic authentication** for gated models on HuggingFace! Simply run `huggingface-cli login` once, and you'll have seamless access to models like Gemma, Llama, and other restricted models. See [Authentication for Gated Models](#authentication-for-gated-models) for details.


## Evaluating a Proprietary Model
### Hardware

The Dockerfile provided in this repository is made to be used on the NVIDIA H100-80GB GPU.
If you would like to run benchmarks on other types of hardware, we invite you to take a look at [these configuration examples](https://github.com/huggingface/optimum-benchmark/tree/main/energy_star) that can be run directly with [Optimum Benchmark](https://github.com/huggingface/optimum-benchmark/). However, evaluations completed on other hardware would not be currently compatable and comparable with the rest of the AI Energy Score data.


### Usage

#### Building the Docker Image

The Docker image includes both optimum-benchmark and ai_energy_benchmarks. Use the provided build script:

```bash
./build.sh
```

Or build manually from the parent directory:

```bash
docker build -f AIEnergyScore/Dockerfile -t ai_energy_score .
```

**Note:** The build must be run from (parent directory) to access both `AIEnergyScore/` and `ai_energy_benchmarks/` directories.

#### Quick Start with Helper Script

For convenience, use the provided helper script that handles all volume mounts automatically:

```bash
cd AIEnergyScore
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b
```

The helper script automatically:
- Runs container as current user
- Mounts HuggingFace cache from `~/.cache/huggingface`
- **Automatically detects and mounts HuggingFace authentication tokens** (for gated models)
- Creates and mounts results directory
- Configures proper environment variables
- Defaults to 20 prompts (customize with `-n` or `--num-samples`)

> **Note for Gated Models**: If you need to access gated models (like `google/gemma-3-4b-pt` or Meta Llama models), run `huggingface-cli login` first. See [Authentication for Gated Models](#authentication-for-gated-models) for details.

**Examples:**
```bash
# Use default 20 samples
./run_docker.sh --config-name text_generation backend.model=openai/gpt-oss-20b

# Test with 100 samples
./run_docker.sh -n 100 --config-name text_generation backend.model=openai/gpt-oss-120b

# Quick test with 5 samples
./run_docker.sh --num-samples 5 --config-name text_generation backend.model=openai/gpt-oss-20b

# Use PyTorch backend with 50 samples
BENCHMARK_BACKEND=pytorch ./run_docker.sh -n 50 --config-name text_generation backend.model=openai/gpt-oss-20b
```

For advanced configuration, see [Docker Volume Mounting Guide](ai_helpers/DOCKER_VOLUME_MOUNTING.md).

#### Authentication for Gated Models

Some models on HuggingFace (e.g., `google/gemma-3-1b-pt`, Meta Llama models) require authentication to access. The `run_docker.sh` script automatically handles authentication using two methods:

**Method 1: HuggingFace CLI Login (Recommended)**

The easiest way to authenticate is using the HuggingFace CLI:

```bash
# One-time setup: login to HuggingFace
huggingface-cli login

# Then use normally - token is automatically mounted
./run_docker.sh --config-name text_generation backend.model=google/gemma-3-1b-pt
```

This creates a token file at `~/.huggingface/token` which is automatically detected and mounted by `run_docker.sh`.

**Method 2: HF_TOKEN Environment Variable**

Alternatively, you can pass your token explicitly:

```bash
# Get your token from https://huggingface.co/settings/tokens
export HF_TOKEN=hf_your_token_here

# Run with token from environment
HF_TOKEN=hf_xxx ./run_docker.sh --config-name text_generation backend.model=google/gemma-3-1b-pt
```

**Troubleshooting Authentication Issues**

If you get a 401 error when accessing gated models:

1. **Verify you have access**: Visit the model page on HuggingFace and accept the terms of use
2. **Check authentication**: Run `huggingface-cli whoami` to verify you're logged in
3. **Verify token file exists**: Check that `~/.huggingface/token` exists
4. **Use diagnostic script**: Run `./ai_helpers/check_hf_auth.sh` to check your authentication status
5. **Use HF_TOKEN**: Try passing the token explicitly via environment variable

The `run_docker.sh` script will display a warning if no authentication is found when you run it.

**Quick Diagnostic**
```bash
# Check your HuggingFace authentication status
./ai_helpers/check_hf_auth.sh
```

#### Manual Usage

Alternatively, you can run your benchmark manually. **Important**: Create the results directory first to avoid permission errors:

```bash
# Create results directory with proper permissions
mkdir -p results

# Run the benchmark
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  ai_energy_score \
  --config-name my_task \
  backend.model=my_model \
  backend.processor=my_processor

#for example
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  -e BENCHMARK_BACKEND=pytorch \
  ai_energy_score \
  --config-name text_generation \
  scenario.num_samples=3 \
  backend.model=openai/gpt-oss-20b

# For gated models, add token file mount and/or HF_TOKEN
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v ~/.huggingface/token:/home/user/.huggingface/token:ro \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  -e HF_TOKEN=hf_your_token_here \
  ai_energy_score \
  --config-name text_generation \
  backend.model=google/gemma-3-1b-pt
```
where `my_task` is the name of a task with a configuration [here](https://github.com/huggingface/optimum-benchmark/tree/main/energy_star), `my_model` is the name of your model that you want to test (which needs to be compatible with either the Transformers or the Diffusers libraries) and `my_processor` is the name of the tokenizer/processor you want to use. In most cases, `backend.model` and `backend.processor` wil lbe identical, except in cases where a model is using another model's tokenizer (e.g. from a LLaMa model).

The rest of the configuration is explained [here](https://github.com/huggingface/optimum-benchmark/)

### Backend Selection

AIEnergyScore supports multiple benchmark backends for flexibility and validation:

| Backend | Tool | Load Generation | Model Location | Use Case |
|---------|------|-----------------|----------------|----------|
| `optimum` (default) | optimum-benchmark | optimum-benchmark generates load | Local GPU (in container) | Official AIEnergyScore benchmarks |
| `pytorch` | ai_energy_benchmarks | ai_energy_benchmarks generates load | Local GPU (in container) | Validation, comparison testing |
| `vllm` | ai_energy_benchmarks | ai_energy_benchmarks generates load | External vLLM server | Production load testing |

#### Default Usage (optimum-benchmark)

```bash
# Standard AIEnergyScore benchmark - run as current user with cache mounting
docker run --gpus all --shm-size 1g \
  --user $(id -u):$(id -g) \
  -v ~/.cache/huggingface:/home/user/.cache/huggingface \
  -v $(pwd)/results:/results \
  -e HOME=/home/user \
  ai_energy_score \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

**Volume mounts:**
- `~/.cache/huggingface:/home/user/.cache/huggingface` - Reuse local HuggingFace model cache (avoids re-downloading)
- `$(pwd)/results:/results` - Persist benchmark results to local directory
- `--user $(id -u):$(id -g)` - Run as current user (not root) for proper file permissions
- `-e HOME=/home/user` - Set HOME environment variable for HuggingFace cache location

#### PyTorch Backend (ai_energy_benchmarks)

```bash
# Use ai_energy_benchmarks with PyTorch backend for validation
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



**Note:** With `BENCHMARK_BACKEND=pytorch`, ai_energy_benchmarks loads the model and generates inference load directly on the GPU, just like optimum-benchmark.

### Running Tests

```bash
cd AIEnergyScore
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt pytest
pytest
```

Use `pytest -m e2e` to run only the end-to-end suites; omit the marker filter to execute the full test collection.

### Batch Testing Multiple Models

The `batch_runner.py` script enables testing multiple models from a CSV configuration file with minimal setup.

#### Quick Start (Docker Backend)

For the PyTorch backend (default), you only need `pandas` installed locally - all AI work runs in Docker:

```bash
cd AIEnergyScore

# Create virtual environment and install minimal dependencies
python3 -m venv .venv
source .venv/bin/activate
pip install pandas

# Run batch tests (uses Docker internally)
python batch_runner.py \
  --model-name "gemma" \
  --output-dir ./results/gemma \
  --num-prompts 10
```

#### Usage Examples

```bash
# Test all models matching "gpt-oss"
python batch_runner.py --model-name "gpt-oss" --num-prompts 20

# Test specific model class (A, B, or C)
python batch_runner.py --class A --num-prompts 50

# Test models with reasoning enabled
python batch_runner.py --reasoning-state "On" --num-prompts 10

# Custom CSV and output directory
python batch_runner.py \
  --csv my_models.csv \
  --output-dir ./custom_results \
  --num-prompts 100
```

#### Command-Line Options

| Option | Description | Default |
|--------|-------------|---------|
| `--csv` | Path to models CSV file | `AI Energy Score (Oct 2025) - Models.csv` |
| `--output-dir` | Output directory for results | `./batch_results` |
| `--backend` | Backend type: `pytorch`, `vllm` | `pytorch` |
| `--num-prompts` | Number of prompts to run | All prompts in dataset |
| `--model-name` | Filter by model name (substring) | - |
| `--class` | Filter by model class (A/B/C) | - |
| `--reasoning-state` | Filter by reasoning state | - |

#### Using vLLM Backend

For the vLLM backend (direct execution), you need the full `ai_energy_benchmarks` package:

```bash
# Install ai_energy_benchmarks from parent directory
pip install -e ../ai_energy_benchmarks[pytorch]
pip install -r requirements.txt

# Run with vLLM backend (requires vLLM server running)
python batch_runner.py \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  --model-name "gpt-oss" \
  --num-prompts 10
```

#### vLLM Backend (ai_energy_benchmarks)

```bash
# Terminal 1: Start vLLM server
vllm serve openai/gpt-oss-120b --port 8000

# Terminal 2: Run benchmark (sends requests to external vLLM server)
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

**Note:** vLLM backend requires a running vLLM server. The benchmark sends HTTP requests to measure energy under production-like serving conditions.

#### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `BENCHMARK_BACKEND` | No | `optimum` | Backend selection: `optimum`, `pytorch`, `vllm` |
| `VLLM_ENDPOINT` | Yes (for vLLM) | - | vLLM server endpoint (e.g., `http://localhost:8000/v1`) |

All backends produce compatible output files (`GPU_ENERGY_WH.txt`, `GPU_ENERGY_SUMMARY.json`) that can be submitted to the AIEnergyScore portal.

> [!WARNING]
> It is essential to adhere to the following GPU usage guidelines:
> - If the model being tested is classified as a Class A or Class B model (generally models with fewer than 66B parameters, depending on quantization and precision settings), testing must be conducted on a single GPU.
> - Running tests on multiple GPUs for these model types will invalidate the results, as it may introduce inconsistencies and misrepresent the model’s actual performance under standard conditions.

Once the benchmarking has been completed, the zipped log files should be uploaded to the [Submission Portal](https://huggingface.co/spaces/AIEnergyScore/submission_portal). The following terms and conditions will need to be accepted upon upload:

*By checking the box below and submitting your energy score data, you confirm and agree to the following:*

1. ***Public Data Sharing**: You consent to the public sharing of the energy performance data derived from your submission. No additional information related to this model including proprietary configurations will be disclosed.*  
2. ***Data Integrity**: You validate that the log files submitted are accurate, unaltered, and generated directly from testing your model as per the specified benchmarking procedures.*  
3. ***Model Representation**: You verify that the model tested and submitted is representative of the production-level version of the model, including its level of quantization and any other relevant characteristics impacting energy efficiency and performance.*

