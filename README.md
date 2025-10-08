![AI Energy Score](/AIEnergyScore_LightBG.png)

Welcome to AI Energy Score! This is an initiative to establish comparable energy efficiency ratings for AI models, helping the industry make informed decisions about sustainability in AI development.

## Key Links
- [Leaderboard](https://huggingface.co/spaces/AIEnergyScore/Leaderboard)
- [FAQ](https://huggingface.github.io/AIEnergyScore/#faq)
- [Documentation](https://huggingface.github.io/AIEnergyScore/#documentation)
- [Label Generator](https://huggingface.co/spaces/AIEnergyScore/Label)


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
docker build -f AIEnergyScore/Dockerfile -t energy_star .
```

**Note:** The build must be run from (parent directory) to access both `AIEnergyScore/` and `ai_energy_benchmarks/` directories.

Then you can run your benchmark with:

```
docker run --gpus all --shm-size 1g energy_star --config-name my_task backend.model=my_model backend.processor=my_processor 
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
# Standard AIEnergyScore benchmark - no changes needed
docker run --gpus all --shm-size 1g energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

#### PyTorch Backend (ai_energy_benchmarks)

```bash
# Use ai_energy_benchmarks with PyTorch backend for validation
docker run --gpus all --shm-size 1g \
  -e BENCHMARK_BACKEND=pytorch \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

**Note:** With `BENCHMARK_BACKEND=pytorch`, ai_energy_benchmarks loads the model and generates inference load directly on the GPU, just like optimum-benchmark.

#### vLLM Backend (ai_energy_benchmarks)

```bash
# Terminal 1: Start vLLM server
vllm serve openai/gpt-oss-120b --port 8000

# Terminal 2: Run benchmark (sends requests to external vLLM server)
docker run --gpus all --shm-size 1g \
  -e BENCHMARK_BACKEND=vllm \
  -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 \
  energy_star \
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

