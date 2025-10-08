#!/usr/bin/env python3
"""
Wrapper script to run ai_energy_benchmarks with optimum-benchmark compatible configs.
Converts optimum-benchmark Hydra config to ai_energy_benchmarks format.
"""

import sys
import json
from pathlib import Path
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_hydra_args(args):
    """Parse Hydra-style arguments (key=value format)"""
    overrides = {}
    config_name = None

    for arg in args:
        if arg.startswith("--config-name"):
            if "=" in arg:
                config_name = arg.split("=", 1)[1]
            else:
                # Next arg is the config name
                idx = args.index(arg)
                if idx + 1 < len(args):
                    config_name = args[idx + 1]
        elif "=" in arg and not arg.startswith("--"):
            key, value = arg.split("=", 1)
            overrides[key] = value

    return config_name, overrides


def load_optimum_config(config_name, config_dir="/optimum-benchmark/energy_star"):
    """Load optimum-benchmark configuration"""
    config_path = Path(config_dir) / f"{config_name}.yaml"

    if not config_path.exists():
        # Try without .yaml extension
        config_path = Path(config_dir) / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    logger.info(f"Loading config from: {config_path}")
    return OmegaConf.load(config_path)


def get_available_gpu_count():
    """Get the number of available GPUs"""
    try:
        import torch
        return torch.cuda.device_count()
    except:
        return 0


def convert_to_ai_energy_benchmarks_config(optimum_config, overrides):
    """Convert optimum-benchmark config to ai_energy_benchmarks format"""

    # Apply overrides to optimum config
    for key, value in overrides.items():
        OmegaConf.update(optimum_config, key, value, merge=True)

    # Extract backend config
    backend = optimum_config.get("backend", {})
    scenario = optimum_config.get("scenario", {})

    # Get device_ids and ensure they're integers
    device_ids = backend.get("device_ids", [0])
    if isinstance(device_ids, str):
        # Handle comma-separated string like "0, 1, 2, 3"
        device_ids = [int(d.strip()) for d in device_ids.split(",")]
    elif isinstance(device_ids, (list, tuple)):
        device_ids = [int(d) for d in device_ids]
    else:
        device_ids = [int(device_ids)]

    # Validate and adjust device_ids based on available GPUs
    available_gpus = get_available_gpu_count()
    if available_gpus > 0:
        # Filter out device IDs that exceed available GPUs
        valid_device_ids = [d for d in device_ids if d < available_gpus]
        if not valid_device_ids:
            # If no valid IDs, use first GPU
            valid_device_ids = [0]
        if valid_device_ids != device_ids:
            logger.warning(f"Adjusted device_ids from {device_ids} to {valid_device_ids} (found {available_gpus} GPUs)")
            device_ids = valid_device_ids
    else:
        logger.warning("No GPUs detected, using device_ids=[0] anyway")
        device_ids = [0]

    # Build ai_energy_benchmarks compatible config
    config = {
        "backend": {
            "type": "pytorch",
            "model": backend.get("model", ""),
            "device": backend.get("device", "cuda"),
            "device_ids": device_ids,
            "torch_dtype": backend.get("torch_dtype", "auto"),
            "device_map": backend.get("device_map", "auto"),
        },
        "dataset": {
            "name": scenario.get("dataset_name", "EnergyStarAI/text_generation"),
            "text_column": scenario.get("text_column_name", "text"),
            "num_samples": int(scenario.get("num_samples", 1000)),
        },
        "generation": {
            "max_new_tokens": int(scenario.get("generate_kwargs", {}).get("max_new_tokens", 100)),
            "min_new_tokens": int(scenario.get("generate_kwargs", {}).get("min_new_tokens", 50)),
        },
        "metrics": {
            "enabled": True,
            "project_name": "ai_energy_benchmark",
        }
    }

    return config


def run_pytorch_backend(config, output_dir):
    """Run ai_energy_benchmarks with PyTorch backend"""
    try:
        from ai_energy_benchmarks.config.parser import (
            BenchmarkConfig, BackendConfig, ScenarioConfig,
            MetricsConfig, ReporterConfig
        )
        from ai_energy_benchmarks.runner import BenchmarkRunner

        logger.info("Initializing ai_energy_benchmarks PyTorch backend")
        logger.info(f"Model: {config['backend']['model']}")
        logger.info(f"Dataset: {config['dataset']['name']}")
        logger.info(f"Samples: {config['dataset']['num_samples']}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build BenchmarkConfig using dataclasses
        backend_cfg = BackendConfig(
            type="pytorch",
            model=config["backend"]["model"],
            device=config["backend"]["device"],
            device_ids=config["backend"]["device_ids"],
        )

        scenario_cfg = ScenarioConfig(
            dataset_name=config["dataset"]["name"],
            text_column_name=config["dataset"]["text_column"],
            num_samples=config["dataset"]["num_samples"],
            generate_kwargs=config["generation"],
        )

        metrics_cfg = MetricsConfig(
            enabled=True,
            type="codecarbon",
            project_name=config["metrics"]["project_name"],
            output_dir=str(output_path / "emissions"),
            country_iso_code="USA",
            region="california",
        )

        reporter_cfg = ReporterConfig(
            type="csv",
            output_file=str(output_path / "benchmark_results.csv"),
        )

        bench_config = BenchmarkConfig(
            name="ai_energy_benchmark",
            backend=backend_cfg,
            scenario=scenario_cfg,
            metrics=metrics_cfg,
            reporter=reporter_cfg,
            output_dir=str(output_path),
        )

        # Create runner
        runner = BenchmarkRunner(bench_config)

        # Validate
        if not runner.validate():
            logger.error("Benchmark validation failed")
            sys.exit(1)

        # Run benchmark
        logger.info("Starting benchmark...")
        results = runner.run()

        # Save results as JSON for compatibility
        result_file = output_path / "benchmark_report.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Benchmark completed successfully")
        logger.info(f"Results saved to: {result_file}")

        if "energy" in results:
            logger.info(f"GPU Energy: {results['energy'].get('gpu_energy_wh', 0):.2f} Wh")

        return results

    except ImportError as e:
        logger.error(f"ai_energy_benchmarks not installed or missing dependencies: {e}")
        logger.error("Install with: pip install ai_energy_benchmarks[pytorch]")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def run_vllm_backend(config, output_dir, endpoint):
    """Run ai_energy_benchmarks with vLLM backend"""
    try:
        from ai_energy_benchmarks.config.parser import (
            BenchmarkConfig, BackendConfig, ScenarioConfig,
            MetricsConfig, ReporterConfig
        )
        from ai_energy_benchmarks.runner import BenchmarkRunner

        logger.info("Initializing ai_energy_benchmarks vLLM backend")
        logger.info(f"Endpoint: {endpoint}")
        logger.info(f"Model: {config['backend']['model']}")

        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Build BenchmarkConfig using dataclasses
        backend_cfg = BackendConfig(
            type="vllm",
            model=config["backend"]["model"],
            endpoint=endpoint,
        )

        scenario_cfg = ScenarioConfig(
            dataset_name=config["dataset"]["name"],
            text_column_name=config["dataset"]["text_column"],
            num_samples=config["dataset"]["num_samples"],
            generate_kwargs=config["generation"],
        )

        metrics_cfg = MetricsConfig(
            enabled=True,
            type="codecarbon",
            project_name=config["metrics"]["project_name"],
            output_dir=str(output_path / "emissions"),
            country_iso_code="USA",
            region="california",
        )

        reporter_cfg = ReporterConfig(
            type="csv",
            output_file=str(output_path / "benchmark_results.csv"),
        )

        bench_config = BenchmarkConfig(
            name="ai_energy_benchmark_vllm",
            backend=backend_cfg,
            scenario=scenario_cfg,
            metrics=metrics_cfg,
            reporter=reporter_cfg,
            output_dir=str(output_path),
        )

        # Create runner
        runner = BenchmarkRunner(bench_config)

        # Validate
        if not runner.validate():
            logger.error("Benchmark validation failed")
            sys.exit(1)

        # Run benchmark
        logger.info("Starting benchmark...")
        results = runner.run()

        # Save results as JSON for compatibility
        result_file = output_path / "benchmark_report.json"
        with open(result_file, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("Benchmark completed successfully")
        logger.info(f"Results saved to: {result_file}")

        if "energy" in results:
            logger.info(f"GPU Energy: {results['energy'].get('gpu_energy_wh', 0):.2f} Wh")

        return results

    except ImportError as e:
        logger.error(f"ai_energy_benchmarks not installed: {e}")
        logger.error("Install with: pip install ai_energy_benchmarks")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error running benchmark: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    import os

    # Parse arguments
    args = sys.argv[1:]
    config_name, overrides = parse_hydra_args(args)

    if not config_name:
        logger.error("No config name provided. Use --config-name <name>")
        sys.exit(1)

    # Get environment variables
    output_dir = os.getenv("RESULTS_DIR", "/results")
    backend_type = os.getenv("BENCHMARK_BACKEND", "pytorch")
    vllm_endpoint = os.getenv("VLLM_ENDPOINT", "")

    # Load and convert config
    logger.info(f"Loading optimum-benchmark config: {config_name}")
    optimum_config = load_optimum_config(config_name)

    logger.info("Converting to ai_energy_benchmarks format")
    config = convert_to_ai_energy_benchmarks_config(optimum_config, overrides)

    # Run appropriate backend
    if backend_type == "vllm":
        if not vllm_endpoint:
            logger.error("VLLM_ENDPOINT environment variable required for vLLM backend")
            sys.exit(1)
        results = run_vllm_backend(config, output_dir, vllm_endpoint)
    else:
        results = run_pytorch_backend(config, output_dir)

    logger.info(f"Results saved to: {output_dir}/benchmark_report.json")


if __name__ == "__main__":
    main()
