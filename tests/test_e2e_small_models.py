from __future__ import annotations

import math
import os
import sys
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
from pathlib import Path
from typing import Dict

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT.parent / "ai_energy_benchmarks"))

from ai_energy_benchmarks.config.parser import (
    BackendConfig,
    BenchmarkConfig,
    MetricsConfig,
    ReporterConfig,
    ScenarioConfig,
)
from ai_energy_benchmarks.runner import BenchmarkRunner


SMALL_MODELS = [
    "sshleifer/tiny-gpt2",
    "distilgpt2",
]

DATASET_NAME = "ag_news"
NUM_SAMPLES = 10


def _project_name(model_id: str) -> str:
    return f"ci_e2e_{model_id.replace('/', '_')}"


def _build_config(model_id: str, output_dir: Path) -> BenchmarkConfig:
    project = _project_name(model_id)
    base_path = output_dir / project

    backend = BackendConfig(
        type="pytorch",
        device="cpu",
        device_ids=[0],
        model=model_id,
        task="text-generation",
    )

    scenario = ScenarioConfig(
        dataset_name=DATASET_NAME,
        text_column_name="text",
        num_samples=NUM_SAMPLES,
        truncation=True,
        reasoning=False,
        generate_kwargs={"max_new_tokens": 8, "min_new_tokens": 1},
    )

    metrics = MetricsConfig(
        enabled=True,
        project_name=project,
        output_dir=str(base_path / "emissions"),
        country_iso_code="USA",
        region="california",
    )

    reporter = ReporterConfig(output_file=str(base_path / "benchmark_results.csv"))

    return BenchmarkConfig(
        name=f"e2e_run_{project}",
        backend=backend,
        scenario=scenario,
        metrics=metrics,
        reporter=reporter,
        output_dir=str(base_path),
    )


@pytest.mark.e2e
def test_small_models_produce_distinct_energy_profiles(tmp_path: Path) -> None:
    energy_readings: Dict[str, float] = {}

    for model_id in SMALL_MODELS:
        config = _build_config(model_id, tmp_path)
        runner = BenchmarkRunner(config)
        results = runner.run()

        assert results["summary"]["total_prompts"] == NUM_SAMPLES
        assert results["summary"]["failed_prompts"] == 0

        energy = results["energy"]["energy_wh"]
        energy_readings[model_id] = energy

    assert len({value for value in energy_readings.values() if not math.isclose(value, 0.0)}) == len(SMALL_MODELS)


@pytest.mark.e2e
@pytest.mark.parametrize("model_id", SMALL_MODELS)
def test_report_outputs_are_written(tmp_path: Path, model_id: str) -> None:
    config = _build_config(model_id, tmp_path)
    runner = BenchmarkRunner(config)
    runner.run()

    base_path = Path(config.output_dir)
    report_file = base_path / "benchmark_results.csv"
    assert report_file.exists()

    energy_dir = base_path / "emissions"
    assert energy_dir.exists()
