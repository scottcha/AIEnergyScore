#!/usr/bin/env python3
"""
Results Aggregator for AI Energy Score Batch Runner.

Aggregates benchmark results from multiple runs into a master CSV file
with comprehensive metrics and derived calculations.
"""

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from model_config_parser import ModelConfig


class ResultsAggregator:
    """Aggregates and saves benchmark results to CSV."""

    # CSV column headers
    COLUMNS = [
        "model_name",
        "model_class",
        "task",
        "reasoning_state",
        "total_prompts",
        "successful_prompts",
        "failed_prompts",
        "total_duration_seconds",
        "avg_total_time",  # Renamed from avg_latency_seconds for clarity
        "avg_time_to_first_token",  # NEW: Time to first token metric
        "total_tokens",
        "total_prompt_tokens",
        "total_completion_tokens",
        "throughput_tokens_per_second",
        "gpu_energy_wh",
        "wh_per_1000_queries",  # NEW: Normalized energy per 1000 queries
        "energy_per_prompt_min_wh",  # NEW: Min energy per prompt
        "energy_per_prompt_max_wh",  # NEW: Max energy per prompt
        "energy_per_prompt_std_wh",  # NEW: Std dev of energy per prompt
        "co2_emissions_g",
        "tokens_per_joule",
        "avg_energy_per_prompt_wh",
        "timestamp",
        "error_message",
    ]

    def __init__(self, output_file: str):
        """Initialize results aggregator.

        Args:
            output_file: Path to master results CSV file
        """
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        # Create file with headers if it doesn't exist
        if not self.output_file.exists():
            self._write_header()

    def _write_header(self) -> None:
        """Write CSV header to file."""
        with open(self.output_file, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writeheader()

    def add_result(
        self,
        config: ModelConfig,
        benchmark_results: Dict[str, Any],
        error_message: Optional[str] = None,
    ) -> None:
        """Add benchmark result to CSV file.

        Args:
            config: ModelConfig for this run
            benchmark_results: Results from BenchmarkRunner
            error_message: Optional error message if run failed
        """
        # Extract summary data
        summary = benchmark_results.get("summary", {})
        energy = benchmark_results.get("energy", {})

        # Calculate derived metrics
        total_prompts = summary.get("total_prompts", 0)
        successful_prompts = summary.get("successful_prompts", 0)
        failed_prompts = summary.get("failed_prompts", 0)
        total_duration = summary.get("total_duration_seconds", 0)
        avg_latency = summary.get("avg_latency_seconds", 0)  # Backend still uses this name
        avg_ttft = summary.get("avg_time_to_first_token", 0)  # NEW: TTFT metric
        total_tokens = summary.get("total_tokens", 0)
        total_prompt_tokens = summary.get("total_prompt_tokens", 0)
        total_completion_tokens = summary.get("total_completion_tokens", 0)
        throughput = summary.get("throughput_tokens_per_second", 0)

        # Energy metrics
        gpu_energy_wh = energy.get("gpu_energy_wh", 0)
        co2_emissions_g = energy.get("emissions_g_co2eq", 0)

        # Calculate tokens per joule (1 Wh = 3600 J)
        tokens_per_joule = 0
        if gpu_energy_wh > 0 and total_tokens > 0:
            energy_joules = gpu_energy_wh * 3600
            tokens_per_joule = total_tokens / energy_joules

        # Calculate average energy per prompt
        avg_energy_per_prompt = 0
        if successful_prompts > 0 and gpu_energy_wh > 0:
            avg_energy_per_prompt = gpu_energy_wh / successful_prompts

        # Calculate Wh per 1000 queries (normalized metric)
        wh_per_1000_queries = 0
        if successful_prompts > 0 and gpu_energy_wh > 0:
            wh_per_1000_queries = (gpu_energy_wh / successful_prompts) * 1000

        # Extract per-prompt energy statistics if available
        energy_per_prompt_min = energy.get("energy_per_prompt_min_wh", 0)
        energy_per_prompt_max = energy.get("energy_per_prompt_max_wh", 0)
        energy_per_prompt_std = energy.get("energy_per_prompt_std_wh", 0)

        # Build row
        row = {
            "model_name": config.model_id,
            "model_class": config.model_class,
            "task": config.task,
            "reasoning_state": config.reasoning_state,
            "total_prompts": total_prompts,
            "successful_prompts": successful_prompts,
            "failed_prompts": failed_prompts,
            "total_duration_seconds": f"{total_duration:.2f}",
            "avg_total_time": f"{avg_latency:.4f}",  # Renamed column
            "avg_time_to_first_token": f"{avg_ttft:.4f}",  # NEW: TTFT metric
            "total_tokens": total_tokens,
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "throughput_tokens_per_second": f"{throughput:.2f}",
            "gpu_energy_wh": f"{gpu_energy_wh:.4f}",
            "wh_per_1000_queries": f"{wh_per_1000_queries:.4f}",
            "energy_per_prompt_min_wh": f"{energy_per_prompt_min:.4f}",
            "energy_per_prompt_max_wh": f"{energy_per_prompt_max:.4f}",
            "energy_per_prompt_std_wh": f"{energy_per_prompt_std:.4f}",
            "co2_emissions_g": f"{co2_emissions_g:.4f}",
            "tokens_per_joule": f"{tokens_per_joule:.4f}",
            "avg_energy_per_prompt_wh": f"{avg_energy_per_prompt:.4f}",
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message or "",
        }

        # Append to CSV
        with open(self.output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

    def add_failed_result(
        self, config: ModelConfig, error_message: str, duration: float = 0
    ) -> None:
        """Add failed result to CSV.

        Args:
            config: ModelConfig for this run
            error_message: Error message
            duration: Duration before failure (optional)
        """
        row = {
            "model_name": config.model_id,
            "model_class": config.model_class,
            "task": config.task,
            "reasoning_state": config.reasoning_state,
            "total_prompts": 0,
            "successful_prompts": 0,
            "failed_prompts": 0,
            "total_duration_seconds": f"{duration:.2f}",
            "avg_total_time": "0.0000",  # Renamed column
            "avg_time_to_first_token": "0.0000",  # NEW: TTFT metric (0 for failed runs)
            "total_tokens": 0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "throughput_tokens_per_second": "0.00",
            "gpu_energy_wh": "0.0000",
            "wh_per_1000_queries": "0.0000",
            "energy_per_prompt_min_wh": "0.0000",
            "energy_per_prompt_max_wh": "0.0000",
            "energy_per_prompt_std_wh": "0.0000",
            "co2_emissions_g": "0.0000",
            "tokens_per_joule": "0.0000",
            "avg_energy_per_prompt_wh": "0.0000",
            "timestamp": datetime.now().isoformat(),
            "error_message": error_message,
        }

        with open(self.output_file, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.COLUMNS)
            writer.writerow(row)

    def get_results_summary(self) -> Dict[str, Any]:
        """Get summary statistics from all results.

        Returns:
            Dict with summary statistics
        """
        if not self.output_file.exists():
            return {}

        total_runs = 0
        successful_runs = 0
        failed_runs = 0
        total_tokens = 0
        total_energy = 0
        total_duration = 0

        with open(self.output_file, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                total_runs += 1
                if not row["error_message"]:
                    successful_runs += 1
                    total_tokens += int(row["total_tokens"])
                    total_energy += float(row["gpu_energy_wh"])
                    total_duration += float(row["total_duration_seconds"])
                else:
                    failed_runs += 1

        summary = {
            "total_runs": total_runs,
            "successful_runs": successful_runs,
            "failed_runs": failed_runs,
            "total_tokens": total_tokens,
            "total_energy_wh": total_energy,
            "total_duration_seconds": total_duration,
        }

        if total_energy > 0 and total_tokens > 0:
            summary["avg_tokens_per_joule"] = total_tokens / (total_energy * 3600)

        if successful_runs > 0:
            summary["avg_energy_per_run"] = total_energy / successful_runs

        return summary


def main():
    """Test the results aggregator."""
    from model_config_parser import ModelConfig

    print("Testing Results Aggregator")
    print("=" * 80)

    # Create aggregator
    aggregator = ResultsAggregator(output_file="./test_results/master_results.csv")

    # Test 1: Add successful result
    print("\n1. Adding successful result...")
    config1 = ModelConfig(
        model_id="openai/gpt-oss-20b",
        priority=1,
        model_class="B",
        task="text_gen",
        reasoning_state="On (High)",
        chat_template="reasoning_effort: high",
    )

    results1 = {
        "summary": {
            "total_prompts": 10,
            "successful_prompts": 10,
            "failed_prompts": 0,
            "total_duration_seconds": 30.5,
            "avg_latency_seconds": 3.05,
            "avg_time_to_first_token": 0.15,  # NEW: TTFT metric
            "total_tokens": 1500,
            "total_prompt_tokens": 500,
            "total_completion_tokens": 1000,
            "throughput_tokens_per_second": 49.18,
        },
        "energy": {
            "gpu_energy_wh": 25.4,
            "emissions_g_co2eq": 12.3,
        },
    }

    aggregator.add_result(config1, results1)
    print("  ✓ Result added")

    # Test 2: Add another result
    print("\n2. Adding another result...")
    config2 = ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1",
        priority=1,
        model_class="C",
        task="text_gen",
        reasoning_state="On",
        chat_template="<think>",
    )

    results2 = {
        "summary": {
            "total_prompts": 10,
            "successful_prompts": 9,
            "failed_prompts": 1,
            "total_duration_seconds": 45.2,
            "avg_latency_seconds": 5.02,
            "avg_time_to_first_token": 0.22,  # NEW: TTFT metric
            "total_tokens": 2000,
            "total_prompt_tokens": 600,
            "total_completion_tokens": 1400,
            "throughput_tokens_per_second": 44.25,
        },
        "energy": {
            "gpu_energy_wh": 35.6,
            "emissions_g_co2eq": 17.2,
        },
    }

    aggregator.add_result(config2, results2)
    print("  ✓ Result added")

    # Test 3: Add failed result
    print("\n3. Adding failed result...")
    config3 = ModelConfig(
        model_id="test/nonexistent-model",
        priority=1,
        model_class="A",
        task="text_gen",
        reasoning_state="Off",
        chat_template="",
    )

    aggregator.add_failed_result(config3, "Model not found", duration=5.0)
    print("  ✓ Failed result added")

    # Test 4: Get summary
    print("\n4. Getting results summary...")
    summary = aggregator.get_results_summary()
    print("  Summary:")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"    {key}: {value:.4f}")
        else:
            print(f"    {key}: {value}")

    print(f"\n✓ Results saved to: {aggregator.output_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
