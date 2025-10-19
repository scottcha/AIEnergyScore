# AI Energy Score Batch Runner

Automated batch runner for AI Energy Score benchmarks. Reads model configurations from CSV and runs comprehensive energy benchmarks with model-specific parameter handling.

## Components

All core modules are in the AIEnergyScore root directory:

### 1. `model_config_parser.py`
Parses the "AI Energy Score (Oct 2025) - Models.csv" file and extracts model configurations including:
- Model HuggingFace ID
- Model class (A/B/C)
- Task type (text_gen, image_gen, etc.)
- Reasoning state
- Chat template parameters

Automatically detects and configures:
- **gpt-oss models**: Harmony formatting with reasoning effort levels
- **DeepSeek models**: `<think>` prefix for thinking mode
- **Qwen models**: `enable_thinking` parameter
- **Hunyuan models**: `/think` prefix
- **EXAONE models**: Inverted reasoning logic
- **Nemotron models**: `/no_think` for reasoning disable
- **Other models**: Generic reasoning parameters

### 2. `parameter_handler.py`
Handles model-specific parameters and prompt formatting:
- Prepares prompts with model-specific prefixes/suffixes
- Builds backend configurations
- Validates model configurations
- Generates safe filenames for results

### 3. `debug_logger.py`
Provides structured logging with file and console output:
- Separate log file for each model run
- Timestamped entries
- Configuration logging
- Prompt-by-prompt progress tracking
- Error details with stack traces
- Results summary

### 4. `results_aggregator.py`
Aggregates results from multiple runs into master CSV:
- Comprehensive metrics (tokens, energy, latency, etc.)
- Derived calculations (tokens/joule, energy/prompt)
- Failed run tracking with error messages
- Summary statistics across all runs

### 5. `batch_runner.py`
Main batch runner script with CLI interface.

### Test Suite (in `ai_helpers/`)
- `test_batch_runner.py`: Component test suite for validation

## Installation

```bash
cd /mnt/storage/src/AIEnergyScore

# Activate virtual environment (or create one)
source .venv/bin/activate

# Install dependencies
pip install pandas

# Install ai_energy_benchmarks if not already installed
cd ../ai_energy_benchmarks
pip install -e .
```

## Usage

### Basic Usage

Run all models from CSV:
```bash
cd /mnt/storage/src/AIEnergyScore
.venv/bin/python batch_runner.py
```

### Filtering Options

Filter by model name (substring match):
```bash
.venv/bin/python batch_runner.py --model-name gpt-oss
```

Filter by model class (A, B, or C):
```bash
.venv/bin/python batch_runner.py --class B
```

Filter by task type:
```bash
.venv/bin/python batch_runner.py --task text_gen
```

Filter by reasoning state:
```bash
.venv/bin/python batch_runner.py --reasoning-state "On (High)"
```

Combine multiple filters:
```bash
.venv/bin/python batch_runner.py --class B --reasoning-state "On"
```

### Configuration Options

Specify number of prompts (for testing):
```bash
.venv/bin/python batch_runner.py --num-prompts 5
```

Use custom output directory:
```bash
.venv/bin/python batch_runner.py --output-dir ./my_results
```

Use PyTorch backend instead of vLLM:
```bash
.venv/bin/python batch_runner.py --backend pytorch
```

Specify vLLM endpoint:
```bash
.venv/bin/python batch_runner.py --endpoint http://192.168.1.100:8000/v1
```

Use custom prompts file:
```bash
.venv/bin/python batch_runner.py --prompts-file ./my_prompts.csv
```

Use custom models CSV:
```bash
.venv/bin/python batch_runner.py --csv ./my_models.csv
```

### Complete Example

Run only Class B gpt-oss models with high reasoning, using 10 prompts:
```bash
.venv/bin/python batch_runner.py \
  --model-name gpt-oss \
  --class B \
  --reasoning-state "High" \
  --num-prompts 10 \
  --output-dir ./gpt_oss_results
```

## Output Structure

```
batch_results/
├── master_results.csv          # Aggregated results from all runs
├── logs/                        # Debug logs
│   ├── openai_gpt-oss-20b_On_High_20251017_120000.log
│   ├── openai_gpt-oss-20b_On_Low_20251017_120530.log
│   └── ...
└── individual_runs/             # Individual benchmark outputs
    ├── openai_gpt-oss-20b_On_High/
    │   ├── benchmark_results.csv
    │   ├── emissions/
    │   └── ...
    ├── openai_gpt-oss-20b_On_Low/
    └── ...
```

## Master Results CSV Columns

- `model_name`: HuggingFace model ID
- `model_class`: Model class (A/B/C)
- `task`: Task type (text_gen, image_gen, etc.)
- `reasoning_state`: Reasoning state from CSV
- `total_prompts`: Total number of prompts attempted
- `successful_prompts`: Number of successful prompts
- `failed_prompts`: Number of failed prompts
- `total_duration_seconds`: Total benchmark duration
- `avg_latency_seconds`: Average latency per prompt
- `total_tokens`: Total tokens generated
- `total_prompt_tokens`: Total input tokens
- `total_completion_tokens`: Total output tokens
- `throughput_tokens_per_second`: Token generation throughput
- `gpu_energy_wh`: GPU energy consumed (Wh)
- `co2_emissions_g`: CO2 emissions (grams)
- `tokens_per_joule`: Energy efficiency metric
- `avg_energy_per_prompt_wh`: Average energy per prompt
- `timestamp`: ISO 8601 timestamp
- `error_message`: Error message if run failed

## Testing

Test individual components:
```bash
cd /mnt/storage/src/AIEnergyScore
.venv/bin/python ai_helpers/test_batch_runner.py
```

Test model config parser standalone:
```bash
.venv/bin/python model_config_parser.py "AI Energy Score (Oct 2025) - Models.csv"
```

Test parameter handler:
```bash
.venv/bin/python parameter_handler.py
```

Test debug logger:
```bash
.venv/bin/python debug_logger.py
```

Test results aggregator:
```bash
.venv/bin/python results_aggregator.py
```

## Model-Specific Handling

### gpt-oss Models
- Auto-enables Harmony formatting
- Extracts reasoning effort from Chat Template
- Passes as `reasoning_params`

Example CSV entry:
```csv
openai/gpt-oss-20b,1,B,text_gen,On (High),"reasoning: True
  reasoning_params:
    reasoning_effort: high"
```

Becomes:
- `use_harmony=True`
- `reasoning_params={"reasoning_effort": "high"}`

### DeepSeek Models
- Detects `<think>` in Chat Template
- Prepends `<think>` to all prompts

Example CSV entry:
```csv
deepseek-ai/DeepSeek-R1,1,C,text_gen,On,"Prepend input with <think>."
```

Becomes:
- `prompt_prefix="<think>"`

### Qwen Models
- Extracts `enable_thinking` parameter
- Passes as reasoning parameter

Example CSV entry:
```csv
Qwen/Qwen3-30B-A3B,1,B,text_gen,On,enable_thinking=True
```

Becomes:
- `reasoning_params={"enable_thinking": True}`

## Troubleshooting

### "No module named 'pandas'"
Install pandas in the virtual environment:
```bash
.venv/bin/pip install pandas
```

### "ai_energy_benchmarks not installed"
Install the benchmarks package:
```bash
cd ../ai_energy_benchmarks
.venv/bin/pip install -e .
```

### "No models match the specified filters"
Check your filter criteria. List all available models:
```bash
.venv/bin/python ai_helpers/model_config_parser.py "AI Energy Score (Oct 2025) - Models.csv"
```

### Backend validation failed
Ensure your backend is running:
- For vLLM: `vllm serve <model> --port 8000`
- For PyTorch: Ensure GPU is available

### Empty results
Check debug logs in `batch_results/logs/` for detailed error information.

## Advanced Usage

### Running in Production
For production runs with all models:
```bash
.venv/bin/python batch_runner.py \
  --output-dir ./production_results_$(date +%Y%m%d) \
  --backend vllm \
  --endpoint http://localhost:8000/v1 \
  > batch_run.log 2>&1
```

### Running Priority 1 Models Only
Use the CSV directly to filter by priority, or filter using grep:
```bash
# Create filtered CSV
grep "^https.*,1," "AI Energy Score (Oct 2025) - Models.csv" > priority1_models.csv

# Run with filtered CSV
.venv/bin/python batch_runner.py --csv priority1_models.csv
```

### Parallel Execution (Future Enhancement)
Currently runs models sequentially. For parallel execution, run multiple instances with different filters:

Terminal 1:
```bash
.venv/bin/python batch_runner.py --class A --output-dir ./results_class_a
```

Terminal 2:
```bash
.venv/bin/python batch_runner.py --class B --output-dir ./results_class_b
```

Then merge the master_results.csv files.

## See Also

- [BATCH_RUNNER_PLAN.md](./BATCH_RUNNER_PLAN.md) - Detailed implementation plan
- [AI Energy Score CSV](../AI%20Energy%20Score%20%28Oct%202025%29%20-%20Models.csv) - Model configurations
- [ai_energy_benchmarks](../../ai_energy_benchmarks/) - Benchmark framework
