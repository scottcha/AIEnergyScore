# Reasoning Parameters Testing

This directory contains comprehensive tests to ensure that reasoning parameters are properly applied and result in different energy consumption across reasoning levels.

## Test Files

### `test_reasoning_parameters.py`
Pytest unit and integration tests for reasoning parameter handling.

**Test Coverage:**
- ✅ Reasoning parameters correctly parsed from CSV
- ✅ Different reasoning levels have different token constraints
- ✅ Docker commands include proper Hydra overrides
- ✅ vLLM backend also respects token constraints
- ✅ Integration test for energy differentiation (marked as `@pytest.mark.integration`)

### `test_reasoning_energy_integration.sh`
Shell script for full integration testing with actual model execution.

**What it does:**
- Runs benchmarks for gpt-oss-20b at all reasoning levels (High, Low, Off)
- Analyzes token counts and energy consumption
- Verifies that different reasoning levels produce different results
- Provides detailed pass/fail analysis

## Running the Tests

### Quick Unit Tests (No Model Execution)
```bash
# Run all unit tests (excludes integration tests)
cd /home/scott/src/AIEnergyScore
.venv/bin/pytest ai_helpers/test_reasoning_parameters.py -v -m "not integration"

# Run only parsing tests
.venv/bin/pytest ai_helpers/test_reasoning_parameters.py::TestReasoningParameterParsing -v

# Run only token constraint tests
.venv/bin/pytest ai_helpers/test_reasoning_parameters.py::TestTokenConstraints -v
```

### Integration Test with Model Execution
```bash
# Run the integration test with 5 prompts (quick test)
cd /home/scott/src/AIEnergyScore
./ai_helpers/test_reasoning_energy_integration.sh 5

# Run with more prompts for stable measurements
./ai_helpers/test_reasoning_energy_integration.sh 20

# Or use pytest (requires uncommenting pytest.skip() in the test)
.venv/bin/pytest ai_helpers/test_reasoning_parameters.py -v -m integration
```

## Expected Results

### Unit Tests
All unit tests should **PASS** immediately:
```
✅ test_csv_file_exists
✅ test_parse_gpt_oss_reasoning_levels
✅ test_reasoning_effort_values
✅ test_off_state_has_no_reasoning_params
✅ test_reasoning_mode_token_limits
✅ test_non_reasoning_mode_token_limits
✅ test_docker_command_includes_reasoning_params
✅ test_vllm_backend_config_token_limits
✅ test_reasoning_params_end_to_end
```

### Integration Test
The integration test should show:

1. **Different token counts** for each reasoning level:
   ```
   Token Generation:
     High:  2500+ tokens  (reasoning allowed to expand)
     Low:   800+ tokens   (less reasoning)
     Off:   10 tokens     (fixed constraint)
   ```

2. **Different energy consumption**:
   ```
   Energy Consumption:
     High:  5.2000 Wh  (highest - more computation)
     Low:   2.1000 Wh  (moderate)
     Off:   0.5000 Wh  (lowest - minimal computation)
   ```

3. **Energy difference** of at least 5% between High and Low

## Troubleshooting

### Problem: Energy values are the same across reasoning levels

**Symptoms:**
```
⚠ WARNING: High and Low have similar energy (1.2% difference)
```

**Possible causes:**
1. Token constraints not applied → Check that docker commands include `scenario.generate_kwargs` overrides
2. Model doesn't respond to reasoning parameters → Verify Harmony format is correctly applied
3. Too few prompts → Increase `--num-prompts` to 20+ for stable measurements

**Debug steps:**
```bash
# Check logs for token generation
grep "Generated.*tokens" test_reasoning_energy_results/logs/*.log

# Verify Harmony format is applied
grep "Using Harmony format" test_reasoning_energy_results/logs/*.log

# Check individual run reports
cat test_reasoning_energy_results/individual_runs/*/benchmark_report.json | grep -A 2 "total_tokens"
```

### Problem: Tests fail with "CSV file not found"

**Solution:**
Make sure you're running tests from the project root:
```bash
cd /home/scott/src/AIEnergyScore
.venv/bin/pytest ai_helpers/test_reasoning_parameters.py -v
```

### Problem: Docker execution fails

**Solution:**
Ensure Docker is running and the `energy_star` image is built:
```bash
docker ps
docker images | grep energy_star

# If image missing, build it
cd /mnt/storage/src/ai_energy_benchmarks
./build_wheel.sh
```

## Test Architecture

### Unit Tests Structure
```
TestReasoningParameterParsing
├── test_csv_file_exists
├── test_parse_gpt_oss_reasoning_levels
├── test_reasoning_effort_values
└── test_off_state_has_no_reasoning_params

TestTokenConstraints
├── test_reasoning_mode_token_limits      (max=8192, min=1)
└── test_non_reasoning_mode_token_limits  (max=10, min=10)

TestDockerCommandConstruction
├── test_docker_command_includes_reasoning_params
└── test_vllm_backend_config_token_limits

TestEnergyDifferentiation (@pytest.mark.integration)
├── test_different_reasoning_levels_produce_different_energy
└── test_token_counts_differ_by_reasoning_level
```

### Token Constraint Logic
The tests verify that `batch_runner.py` applies these constraints:

**Reasoning Mode (On - High/Medium/Low):**
```python
max_new_tokens = 8192  # Very high limit
min_new_tokens = 1     # Allow short responses
```

**Non-Reasoning Mode (Off/N/A):**
```python
max_new_tokens = 10    # Fixed short response
min_new_tokens = 10    # Ensure consistency
```

## CI/CD Integration

These tests can be integrated into CI/CD pipelines:

```bash
# Quick validation (unit tests only)
pytest ai_helpers/test_reasoning_parameters.py -v -m "not integration"

# Full validation (includes model execution)
pytest ai_helpers/test_reasoning_parameters.py -v -m integration

# Or use the shell script
./ai_helpers/test_reasoning_energy_integration.sh 5
```

## Maintenance

When modifying reasoning parameter handling:

1. **Run unit tests immediately** to catch regressions
2. **Update expected values** if token constraints change
3. **Run integration test** to verify energy differentiation still works
4. **Update this README** if test behavior changes

## Related Files

- `/home/scott/src/AIEnergyScore/batch_runner.py` - Main batch runner (lines 139-155, 498-512)
- `/home/scott/src/AIEnergyScore/model_config_parser.py` - CSV parsing and reasoning param extraction
- `/home/scott/src/AIEnergyScore/parameter_handler.py` - Parameter validation and handling
- `/mnt/storage/src/ai_energy_benchmarks/ai_energy_benchmarks/backends/pytorch.py` - Reasoning parameter application
