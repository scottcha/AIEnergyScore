# AI Energy Score Batch Runner - Quick Start

## Prerequisites

```bash
cd /mnt/storage/src/AIEnergyScore
source .venv/bin/activate

# Ensure dependencies are installed
pip install pandas
cd ../ai_energy_benchmarks && pip install -e . && cd ../AIEnergyScore
```

**Notes**:
- The batch runner uses **PyTorch backend by default**. Use `--backend vllm` if you have a vLLM server running.
- Uses HuggingFace dataset `scottcha/reasoning_text_generation` by default. Use `--prompts-file` to specify a custom file.

## Common Commands

### 1. Test with a Single Model (Recommended First Step)

```bash
# Test with gpt-oss-20b, high reasoning, 3 prompts
.venv/bin/python batch_runner.py \
  --model-name "gpt-oss-20b" \
  --reasoning-state "High" \
  --num-prompts 3 \
  --output-dir ./test_run
```

### 2. Run All gpt-oss Models

```bash
# Uses PyTorch backend by default
.venv/bin/python batch_runner.py \
  --model-name gpt-oss \
  --num-prompts 3 \
  --output-dir ./gpt_oss_results
```

### 3. Run Class A Models (Smaller Models)

```bash
.venv/bin/python batch_runner.py \
  --class A \
  --num-prompts 10 \
  --output-dir ./class_a_results
```

### 4. Run Class B Models (Medium Models)

```bash
.venv/bin/python batch_runner.py \
  --class B \
  --num-prompts 10 \
  --output-dir ./class_b_results
```

### 5. Run Class C Models (Large Models)

```bash
.venv/bin/python batch_runner.py \
  --class C \
  --num-prompts 5 \
  --output-dir ./class_c_results
```

### 6. Run Models with High Reasoning Only

```bash
.venv/bin/python batch_runner.py \
  --reasoning-state "High" \
  --output-dir ./high_reasoning_results
```

### 7. Run All Models (Full Benchmark)

```bash
.venv/bin/python batch_runner.py \
  --output-dir ./full_results_$(date +%Y%m%d_%H%M%S)
```

## Quick Checks

### View Available Models

```bash
.venv/bin/python model_config_parser.py "AI Energy Score (Oct 2025) - Models.csv"
```

### Test Components

```bash
.venv/bin/python ai_helpers/test_batch_runner.py
```

### View Help

```bash
.venv/bin/python batch_runner.py --help
```

## Check Results

### View Master Results CSV

```bash
# View all results
cat batch_results/master_results.csv

# View with column headers nicely formatted
column -t -s',' batch_results/master_results.csv | less -S

# Count successful/failed runs
tail -n +2 batch_results/master_results.csv | awk -F',' '{if ($19 == "") print "success"; else print "failed"}' | sort | uniq -c
```

### View Debug Logs

```bash
# List all log files
ls -lh batch_results/logs/

# View a specific log
cat batch_results/logs/openai_gpt-oss-20b_On_High_*.log

# View most recent log
ls -t batch_results/logs/*.log | head -1 | xargs cat
```

### Check Individual Run Results

```bash
# List individual runs
ls batch_results/individual_runs/

# View benchmark results for a specific run
cat batch_results/individual_runs/openai_gpt-oss-20b_On_High/benchmark_results.csv
```

## Troubleshooting

### Using vLLM Backend

PyTorch is the default backend. To use vLLM, start a vLLM server and specify the backend:
```bash
# Terminal 1: Start vLLM server
vllm serve openai/gpt-oss-20b --port 8000

# Terminal 2: Run batch with vLLM backend
.venv/bin/python batch_runner.py --backend vllm --endpoint http://localhost:8000/v1
```

### PyTorch Backend (Default)

No additional setup needed - PyTorch backend is used by default:
```bash
.venv/bin/python batch_runner.py  # Uses PyTorch automatically
```

### Custom vLLM Endpoint

```bash
.venv/bin/python batch_runner.py --endpoint http://192.168.1.100:8000/v1
```

### Missing Pandas

```bash
.venv/bin/pip install pandas
```

### Check What Models Would Run

Use the parser to see what matches your filters:
```bash
.venv/bin/python -c "
from model_config_parser import ModelConfigParser
parser = ModelConfigParser('AI Energy Score (Oct 2025) - Models.csv')
configs = parser.parse()
filtered = parser.filter_configs(configs, model_name='gpt-oss')
for c in filtered:
    print(f'{c.model_id} - {c.reasoning_state}')
"
```

## Filter Combinations

### gpt-oss Models with High Reasoning

```bash
.venv/bin/python batch_runner.py \
  --model-name gpt-oss \
  --reasoning-state "High"
```

### Class B Models with Any Reasoning On

```bash
.venv/bin/python batch_runner.py \
  --class B \
  --reasoning-state "On"
```

### Specific Model, All Reasoning States

```bash
.venv/bin/python batch_runner.py \
  --model-name "DeepSeek-R1"
```

### Text Generation Models Only

```bash
.venv/bin/python batch_runner.py \
  --task text_gen
```

## Output Structure

After running, you'll have:

```
batch_results/
├── master_results.csv          # ← Main results file
├── logs/                        # ← Check here if issues
│   └── *.log
└── individual_runs/             # ← Detailed per-model results
    └── */
```

## Key Metrics in CSV

- `tokens_per_joule` - Energy efficiency (higher = better)
- `avg_energy_per_prompt_wh` - Energy cost per prompt (lower = better)
- `throughput_tokens_per_second` - Generation speed
- `gpu_energy_wh` - Total energy used
- `co2_emissions_g` - Carbon emissions

## Tips

1. **Start small**: Use `--num-prompts 3` for initial testing
2. **Filter wisely**: Run subsets before full batch
3. **Check logs**: Debug logs show every step
4. **Monitor resources**: Watch GPU memory and disk space
5. **Save outputs**: Use timestamped output directories
6. **Test filters**: Use model_config_parser.py to preview what will run

## Example Workflow

```bash
# 1. Test setup with one model
.venv/bin/python batch_runner.py \
  --model-name "gpt-oss-20b" \
  --reasoning-state "High" \
  --num-prompts 3

# 2. Check results
cat batch_results/master_results.csv
cat batch_results/logs/*.log | tail -50

# 3. Run small batch (Class A)
.venv/bin/python batch_runner.py \
  --class A \
  --num-prompts 10 \
  --output-dir ./class_a_$(date +%Y%m%d)

# 4. If successful, run full batch
.venv/bin/python batch_runner.py \
  --output-dir ./full_batch_$(date +%Y%m%d)
```

## More Information

- Full documentation: [BATCH_RUNNER_README.md](BATCH_RUNNER_README.md)
- Implementation details: [BATCH_RUNNER_PLAN.md](BATCH_RUNNER_PLAN.md)
- Summary: [BATCH_RUNNER_SUMMARY.md](BATCH_RUNNER_SUMMARY.md)
