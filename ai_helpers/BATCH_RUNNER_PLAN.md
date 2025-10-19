# AI Energy Score Batch Runner Implementation Plan

## Overview
Create a batch runner that reads the "AI Energy Score (Oct 2025) - Models.csv" file and runs benchmarks for each model in sequence, handling special formatting requirements (Harmony for gpt-oss models, reasoning parameters for others), with filtering capabilities and comprehensive results tracking.

## Key Components to Create

### 1. **CSV Model Configuration Parser** (`ai_helpers/model_config_parser.py`)
- Parse the CSV file to extract:
  - Model HuggingFace ID (column: "Models to Add")
  - Priority (column: "Priority (1=hi)")
  - Class (A/B/C - column: "Class")
  - Task (column: "Task")
  - Reasoning State (column: "Reasoning State")
  - Chat Template/Parameters (column: "Chat Template")
- Parse the chat template column to extract:
  - For gpt-oss models: `reasoning_effort` (low/medium/high) + enable Harmony formatting
  - For other models: Extract reasoning parameters or prefix text
- Return list of model configurations ready for benchmarking

### 2. **Batch Runner Script** (`AIEnergyScore/batch_runner.py`)
- Command-line interface with filters:
  - `--model-name`: Filter by specific model name (substring match)
  - `--class`: Filter by model class (A, B, or C)
  - `--task`: Filter by task type (e.g., text_gen, image_gen)
  - `--reasoning-state`: Filter by reasoning state (e.g., "On", "Off", "On (High)")
  - `--num-prompts`: Number of prompts to run (default: all prompts from prompts.csv)
  - `--output-dir`: Output directory for results (default: `./batch_results`)
  - `--backend`: Backend type (vllm or pytorch, default: vllm)
  - `--endpoint`: vLLM endpoint URL (default: http://localhost:8000/v1)
  - `--prompts-file`: Path to prompts CSV file (default: ../ai_energy_benchmarks/prompts.csv)

### 3. **Model-Specific Parameter Handler** (`ai_helpers/parameter_handler.py`)
- Detect model type and apply appropriate formatting:
  - **gpt-oss models**: Enable `use_harmony=True` + parse reasoning_effort from Chat Template
  - **DeepSeek models**: Prepend `<think>` tag for thinking mode
  - **Qwen models**: Use `enable_thinking=True/False` parameter
  - **Other models with reasoning**: Apply as prefix to prompt
- Build `reasoning_params` dict for backend

### 4. **Results Aggregator** (`ai_helpers/results_aggregator.py`)
- Aggregate results from multiple runs into a master CSV:
  - Model name (HuggingFace ID)
  - Model class (A/B/C)
  - Task type
  - Reasoning state
  - Total prompts
  - Successful prompts
  - Failed prompts
  - Total duration (seconds)
  - Average latency per prompt (seconds)
  - Total tokens generated
  - Total prompt tokens
  - Total completion tokens
  - Throughput (tokens/second)
  - GPU energy (Wh) - if available
  - CO2 emissions (g) - if available
  - Tokens per joule (efficiency metric)
  - Average energy per prompt (Wh/prompt)
  - Timestamp

### 5. **Debug Logger** (`ai_helpers/debug_logger.py`)
- Separate debug log file for each model run
- Log format: `{model_name}_{reasoning_state}_{timestamp}.log`
- Capture:
  - Configuration details
  - Each prompt processing step
  - Errors and warnings
  - Harmony formatting application (if applicable)
  - Backend responses
  - Performance metrics

## Implementation Steps

### Step 1: Create Model Config Parser
- Read CSV file with pandas
- Clean and validate data
- Parse Chat Template column to extract parameters
- Handle special cases (gpt-oss, DeepSeek, Qwen, etc.)
- Return structured list of model configs

### Step 2: Create Parameter Handler
- Detect model type from HuggingFace ID
- Build appropriate reasoning_params dict
- Handle Harmony formatting flag for gpt-oss
- Apply prefix/suffix transformations for non-gpt-oss models

### Step 3: Create Batch Runner
- Parse command-line arguments with filtering
- Load model configurations from CSV
- Apply filters (model name, class, task, reasoning state)
- For each filtered model:
  - Setup output directory for this model
  - Configure backend with model-specific parameters
  - Create BenchmarkConfig with appropriate settings
  - Run BenchmarkRunner
  - Capture results
  - Log to debug file
  - Append to master results CSV

### Step 4: Create Results Aggregator
- Collect results from each benchmark run
- Calculate derived metrics (tokens/joule, energy/prompt)
- Write to master CSV with all columns
- Handle missing data gracefully

### Step 5: Create Debug Logger
- Setup file handler per model
- Format log messages with timestamps
- Capture stdout/stderr from benchmark runs
- Store in organized directory structure

## File Structure
```
AIEnergyScore/
├── ai_helpers/
│   ├── model_config_parser.py    # Parse CSV and extract configs
│   ├── parameter_handler.py       # Handle model-specific parameters
│   ├── results_aggregator.py      # Aggregate results to CSV
│   └── debug_logger.py            # Debug logging utilities
├── batch_runner.py                # Main batch runner script
├── batch_results/                 # Output directory (created)
│   ├── master_results.csv         # Aggregated results
│   ├── logs/                      # Debug logs per model
│   └── individual_runs/           # Individual benchmark outputs
└── AI Energy Score (Oct 2025) - Models.csv  # Input config
```

## Key Design Decisions

1. **Harmony Formatting**: Auto-detected based on model name (gpt-oss), applied via `use_harmony=True` in backends
2. **Reasoning Parameters**: Parsed from CSV "Chat Template" column, passed as `reasoning_params` dict
3. **Non-gpt-oss Reasoning**: Applied as prompt prefix for simplicity (can be enhanced later)
4. **Filtering**: Multiple independent filters (AND logic) for flexibility
5. **Progress Tracking**: Print status for each model + detailed debug logs
6. **Error Handling**: Continue on failure, log errors, mark as failed in results
7. **Prompts**: Use existing ai_energy_benchmarks/prompts.csv, with configurable num_samples

## Testing Strategy
- Test with single model first (gpt-oss-20b with high reasoning)
- Test filtering functionality
- Validate Harmony formatting application
- Test error handling with non-existent model
- Run full batch on subset of models
- Verify results CSV format and completeness

## Model-Specific Handling Details

### gpt-oss Models (openai/gpt-oss-20b, openai/gpt-oss-120b)
- **Detection**: Model name contains "gpt-oss"
- **Harmony Formatting**: Enable via `use_harmony=True` in backend
- **Reasoning Effort**: Extract from Chat Template column
  - "reasoning_effort: high" → `reasoning_params={"reasoning_effort": "high"}`
  - "reasoning_effort: low" → `reasoning_params={"reasoning_effort": "low"}`
  - "reasoning_effort: medium" → `reasoning_params={"reasoning_effort": "medium"}`
- **Example CSV Parsing**:
  - Row: "openai/gpt-oss-20b", "On (High)", "reasoning: True\n  reasoning_params:\n    reasoning_effort: high"
  - Result: `use_harmony=True`, `reasoning_params={"reasoning_effort": "high"}`

### DeepSeek Models (deepseek-ai/DeepSeek-R1, DeepSeek-R1-0528)
- **Detection**: Model name contains "DeepSeek"
- **Thinking Mode**: Chat Template says "Prepend input with <think>"
- **Implementation**: Add "<think>" as prompt prefix
- **Example**:
  - Original prompt: "What is quantum computing?"
  - Modified: "<think>What is quantum computing?"

### Qwen Models (Qwen/Qwen3-30B-A3B, Qwen/Qwen3-0.6B)
- **Detection**: Model name contains "Qwen"
- **Thinking Parameter**: Chat Template says "enable_thinking=True/False"
- **Implementation**: Pass as reasoning parameter
- **Example CSV Parsing**:
  - Chat Template: "enable_thinking=True"
  - Result: `reasoning_params={"enable_thinking": True}`

### Hunyuan Models (tencent/Hunyuan-1.8B-Instruct)
- **Detection**: Model name contains "Hunyuan"
- **Slow-Thinking Mode**: Chat Template says "/think before the prompt"
- **Implementation**: Add "/think" as prompt prefix
- **Example**:
  - Original prompt: "Explain AI"
  - Modified: "/think Explain AI"

### EXAONE Models (LGAI-EXAONE/EXAONE-4.0-32B)
- **Detection**: Model name contains "EXAONE"
- **Note**: Inverted logic - "Off" means enable_thinking=True
- **Implementation**: Parse enable_thinking value carefully

### Nemotron Models (nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)
- **Detection**: Model name contains "Nemotron"
- **Reasoning OFF Mode**: Chat Template says "/no_think in the system prompt"
- **Implementation**: Modify system prompt or add prefix
- **Default**: Reasoning is ON by default

### Standard Models (No special reasoning)
- **Detection**: Chat Template says "N/A (default)" or is empty
- **Implementation**: No special handling, standard inference
- **Examples**: mistralai/Mistral-Nemo-Instruct-2407, ibm-granite/granite-4.0-micro

## Error Handling

### Model Loading Failures
- Log error to debug file
- Mark run as failed in results CSV
- Continue to next model
- Include error message in results

### Inference Failures
- Log individual prompt failures
- Continue with remaining prompts
- Record success/failure counts
- Include partial results in CSV

### Configuration Parsing Errors
- Validate CSV structure on load
- Warn about malformed Chat Template entries
- Use sensible defaults when possible
- Skip unparseable rows with warning

## Output Format

### Master Results CSV Columns
```csv
model_name,model_class,task,reasoning_state,total_prompts,successful_prompts,failed_prompts,total_duration_seconds,avg_latency_seconds,total_tokens,total_prompt_tokens,total_completion_tokens,throughput_tokens_per_second,gpu_energy_wh,co2_emissions_g,tokens_per_joule,avg_energy_per_prompt_wh,timestamp,error_message
```

### Debug Log Format
```
[2025-10-17 14:30:00] INFO: Starting benchmark for openai/gpt-oss-20b
[2025-10-17 14:30:00] INFO: Model class: B, Task: text_gen, Reasoning: On (High)
[2025-10-17 14:30:00] INFO: Harmony formatting: ENABLED
[2025-10-17 14:30:00] INFO: Reasoning params: {'reasoning_effort': 'high'}
[2025-10-17 14:30:01] INFO: Processing prompt 1/10: "What is machine learning?"
[2025-10-17 14:30:02] INFO: Prompt 1 completed in 1.2s (150 tokens)
[2025-10-17 14:30:02] INFO: Processing prompt 2/10: "Explain photosynthesis..."
...
[2025-10-17 14:30:30] INFO: Benchmark completed successfully
[2025-10-17 14:30:30] INFO: Total duration: 30.5s, Successful: 10/10
[2025-10-17 14:30:30] INFO: Energy: 25.4 Wh, CO2: 12.3 g
```

## CLI Examples

### Run all models
```bash
python batch_runner.py --backend vllm --endpoint http://localhost:8000/v1
```

### Run only gpt-oss models
```bash
python batch_runner.py --model-name gpt-oss --backend vllm
```

### Run only Class B models with high reasoning
```bash
python batch_runner.py --class B --reasoning-state "On (High)"
```

### Run specific model with limited prompts for testing
```bash
python batch_runner.py --model-name "gpt-oss-20b" --num-prompts 5
```

### Run with custom output directory
```bash
python batch_runner.py --output-dir ./my_results --backend pytorch
```

## Next Steps After Implementation

1. **Validation**: Run test suite to verify all model types handled correctly
2. **Documentation**: Add usage examples and troubleshooting guide
3. **Enhancement**: Add parallel execution support for multiple GPUs
4. **Integration**: Connect with existing AIEnergyScore infrastructure
5. **Monitoring**: Add real-time progress dashboard
