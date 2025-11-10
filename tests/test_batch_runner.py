#!/usr/bin/env python3
"""
Test script for batch runner components.

Tests individual modules before running full batch.
"""

import sys
from pathlib import Path

# Add path for imports - use parent directory
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_model_config_parser():
    """Test model config parser."""
    print("=" * 80)
    print("TEST 1: Model Config Parser")
    print("=" * 80)

    from model_config_parser import ModelConfigParser

    csv_path = Path(__file__).parent.parent / "oct_2025_models.csv"

    assert csv_path.exists(), f"CSV file not found: {csv_path}"

    parser = ModelConfigParser(str(csv_path))
    configs = parser.parse()

    assert len(configs) > 0, "No configurations parsed"
    print(f"✓ Parsed {len(configs)} configurations")

    # Show first few
    for i, config in enumerate(configs[:3], 1):
        print(f"\n  Model {i}: {config.model_id}")
        print(f"    Class: {config.model_class}, Task: {config.task}")
        print(f"    Reasoning State: {config.reasoning_state}")
        print(f"    Use Harmony: {config.use_harmony}")
        print(f"    Reasoning Params: {config.reasoning_params}")
        print(f"    Prompt Prefix: '{config.prompt_prefix}'")

    # Test filtering
    print("\n  Testing filters:")
    gpt_oss = parser.filter_configs(configs, model_name="gpt-oss")
    print(f"    gpt-oss models: {len(gpt_oss)}")
    assert len(gpt_oss) > 0, "No gpt-oss models found"

    class_b = parser.filter_configs(configs, model_class="B")
    print(f"    Class B models: {len(class_b)}")
    assert len(class_b) > 0, "No Class B models found"

    reasoning_high = parser.filter_configs(configs, reasoning_state="High")
    print(f"    High reasoning models: {len(reasoning_high)}")

    print("\n✓ Model Config Parser: PASSED")


def test_parameter_handler():
    """Test parameter handler."""
    print("\n" + "=" * 80)
    print("TEST 2: Parameter Handler")
    print("=" * 80)

    from model_config_parser import ModelConfig
    from parameter_handler import ParameterHandler

    # Test gpt-oss
    config1 = ModelConfig(
        model_id="openai/gpt-oss-20b",
        priority=1,
        model_class="B",
        task="text_gen",
        reasoning_state="On (High)",
        chat_template="reasoning_effort: high",
        use_harmony=True,
        reasoning_params={"reasoning_effort": "high"},
    )

    backend_cfg = ParameterHandler.get_backend_config(config1)
    gen_kwargs = ParameterHandler.get_generation_kwargs(
        config1, {"max_tokens": 100}
    )
    filename = ParameterHandler.format_model_name_for_filename(
        config1.model_id, config1.reasoning_state
    )

    print(f"  gpt-oss Backend Config: {backend_cfg}")
    print(f"  Generation Kwargs: {gen_kwargs}")
    print(f"  Filename: {filename}")

    assert backend_cfg["use_harmony"] is True, "Harmony should be enabled for gpt-oss"
    assert "reasoning_params" in gen_kwargs, "Reasoning params should be in gen_kwargs"

    # Test DeepSeek
    config2 = ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1",
        priority=1,
        model_class="C",
        task="text_gen",
        reasoning_state="On",
        chat_template="<think>",
        prompt_prefix="<think>",
    )

    prompt = ParameterHandler.prepare_prompt(config2, "Explain AI")
    print(f"\n  DeepSeek Prompt: {prompt}")
    assert prompt.startswith("<think>"), "DeepSeek prompt should start with <think>"

    # Test validation
    is_valid, error = ParameterHandler.validate_config(config1)
    print(f"\n  Validation: {is_valid}, Error: {error}")
    assert is_valid is True, f"Config validation failed: {error}"

    print("\n✓ Parameter Handler: PASSED")


def test_debug_logger():
    """Test debug logger."""
    print("\n" + "=" * 80)
    print("TEST 3: Debug Logger")
    print("=" * 80)

    from debug_logger import DebugLogger

    logger = DebugLogger(
        log_dir="./test_logs",
        model_name="test/model",
        reasoning_state="On",
        console_level=30,  # WARNING - reduce console output
    )

    logger.log_benchmark_start("test/model", 5, "vllm")
    logger.log_config({"test": "value"})
    logger.log_prompt_start(1, 5, "Test prompt")
    logger.log_prompt_end(1, True, 1.5, 100)
    logger.log_benchmark_end(True, 10.0, 5, 5)

    log_file = logger.get_log_file_path()
    logger.close()

    assert Path(log_file).exists(), f"Log file not created: {log_file}"
    print(f"  ✓ Log file created: {log_file}")
    print("\n✓ Debug Logger: PASSED")


def test_results_aggregator():
    """Test results aggregator."""
    print("\n" + "=" * 80)
    print("TEST 4: Results Aggregator")
    print("=" * 80)

    from model_config_parser import ModelConfig
    from results_aggregator import ResultsAggregator

    aggregator = ResultsAggregator("./test_results/test_master.csv")

    config = ModelConfig(
        model_id="test/model",
        priority=1,
        model_class="B",
        task="text_gen",
        reasoning_state="On",
        chat_template="",
    )

    results = {
        "summary": {
            "total_prompts": 10,
            "successful_prompts": 10,
            "failed_prompts": 0,
            "total_duration_seconds": 30.0,
            "avg_latency_seconds": 3.0,
            "total_tokens": 1000,
            "total_prompt_tokens": 300,
            "total_completion_tokens": 700,
            "throughput_tokens_per_second": 33.33,
        },
        "energy": {
            "gpu_energy_wh": 20.0,
            "emissions_g_co2eq": 10.0,
        },
    }

    aggregator.add_result(config, results)

    summary = aggregator.get_results_summary()
    print(f"  ✓ Results added, summary: {summary}")

    assert summary["total_runs"] > 0, "No runs recorded"
    assert summary["successful_runs"] > 0, "No successful runs"

    print("\n✓ Results Aggregator: PASSED")


def main():
    """Run all tests when script is run directly (not via pytest)."""
    print("\n" + "=" * 80)
    print("BATCH RUNNER COMPONENT TESTS")
    print("=" * 80 + "\n")

    # Run tests and catch any assertion errors
    test_results = []
    tests = [
        ("Model Config Parser", test_model_config_parser),
        ("Parameter Handler", test_parameter_handler),
        ("Debug Logger", test_debug_logger),
        ("Results Aggregator", test_results_aggregator),
    ]

    for name, test_func in tests:
        try:
            test_func()
            test_results.append((name, True))
        except AssertionError as e:
            print(f"\n✗ {name}: FAILED - {e}")
            test_results.append((name, False))
        except Exception as e:
            print(f"\n✗ {name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
            test_results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, result in test_results if result)
    total = len(test_results)

    for name, result in test_results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"  {name:30s} {status}")

    print("=" * 80)
    print(f"\nTotal: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All tests passed! Ready to run batch runner.")
        return 0
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
