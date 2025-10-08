# Backend Switching Strategy: optimum-benchmark ↔ ai_energy_benchmarks

**Version:** 1.0
**Date:** 2025-10-07
**Status:** Design Document
**Context:** Post-POC validation showing PyTorch backend produces comparable results to optimum-benchmark

## Executive Summary

This document outlines the strategy for enabling flexible switching between two benchmark backends:
1. **optimum-benchmark** (AIEnergyScore's current dependency - HuggingFace PyTorch backend)
2. **ai_energy_benchmarks** (NeuralWatt's custom implementation - PyTorch + vLLM backends)

The POC phase has validated that the ai_energy_benchmarks PyTorch backend produces energy and performance metrics comparable to optimum-benchmark. This design enables:
- AIEnergyScore to use either backend without code changes
- neuralwatt_cloud to leverage ai_energy_benchmarks as a library
- Standalone ai_energy_benchmarks usage
- Future extensibility for additional backends (vLLM, Dynamo, SGLang)

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Backend Switching Architecture](#2-backend-switching-architecture)
3. [ai_energy_benchmarks Package Enhancements](#3-ai_energy_benchmarks-package-enhancements)
4. [Integration Strategies](#4-integration-strategies)
5. [Configuration Management](#5-configuration-management)
6. [Implementation Roadmap](#6-implementation-roadmap)
7. [Testing & Validation Strategy](#7-testing--validation-strategy)
8. [Decision Points & Recommendations](#8-decision-points--recommendations)
9. [Risk Assessment](#9-risk-assessment)
10. [Appendices](#10-appendices)

---

## 1. Current State Analysis

### 1.1 AIEnergyScore (Current Implementation)

**Architecture:**
```
AIEnergyScore/
├── Dockerfile                    # PyTorch 2.8.0-cuda12.9 base
├── entrypoint.sh                 # Calls optimum-benchmark
├── check_h100.py                 # GPU validation
├── summarize_gpu_wh.py           # Energy summarization
├── text_generation_validation.yaml  # Hydra config
└── pytorch_validation.yaml       # Hydra config
```

**Key Characteristics:**
- Uses HuggingFace's **optimum-benchmark** package
- Configuration via Hydra/OmegaConf (`.yaml` files)
- Validated on NVIDIA H100-80GB GPU
- PyTorch backend only (via optimum-benchmark)
- Docker-based execution
- Energy measurement via optimum-benchmark's built-in trackers
- Results format: `benchmark_report.json` → GPU_ENERGY_WH extraction

**Current Flow:**
```bash
docker run --gpus all energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b

→ entrypoint.sh
  → check_h100.py (validate GPU)
  → optimum-benchmark --config-dir /optimum-benchmark/energy_star/ [args]
  → summarize_gpu_wh.py (extract GPU_ENERGY_WH)
```

**Dependencies:**
- `optimum-benchmark` (git clone from HuggingFace repo)
- `torch`, `transformers`, `datasets`
- Hydra configuration system
- CodeCarbon (via optimum-benchmark)

### 1.2 ai_energy_benchmarks (POC Implementation)

**Architecture:**
```
ai_energy_benchmarks/
├── ai_energy_benchmarks/
│   ├── backends/
│   │   ├── pytorch.py           # PyTorch backend (VALIDATED)
│   │   └── vllm.py              # vLLM backend (VALIDATED)
│   ├── config/
│   │   └── parser.py            # Hydra config parser
│   ├── datasets/
│   │   └── huggingface.py       # HF dataset loader
│   ├── metrics/
│   │   └── codecarbon.py        # CodeCarbon metrics
│   ├── reporters/
│   │   └── csv_reporter.py      # CSV output
│   └── runner.py                # Main benchmark runner
├── configs/
│   ├── backend/
│   │   ├── pytorch.yaml         # PyTorch config
│   │   └── vllm.yaml            # vLLM config
│   └── pytorch_validation.yaml  # Full config
├── run_benchmark.sh             # Shell wrapper
└── pyproject.toml               # Package metadata
```

**Key Characteristics:**
- Supports **PyTorch** and **vLLM** backends
- Hydra/OmegaConf configuration (compatible with AIEnergyScore)
- Modular plugin architecture (backends, metrics, reporters)
- CodeCarbon for energy measurement (GPU, CPU, RAM, CO₂)
- Python package with CLI entry point
- POC validated on NVIDIA RTX Pro 6000

**Current Flow:**
```bash
./run_benchmark.sh configs/pytorch_validation.yaml

→ python -c "from ai_energy_benchmarks.runner import run_benchmark_from_config"
  → ConfigParser.load_config()
  → BenchmarkRunner.__init__()
    → _initialize_backend() (PyTorchBackend or VLLMBackend)
    → _initialize_dataset() (HuggingFaceDataset)
    → _initialize_metrics() (CodeCarbonCollector)
    → _initialize_reporter() (CSVReporter)
  → BenchmarkRunner.run()
    → validate()
    → load dataset
    → metrics.start()
    → backend.run_inference() (loop over prompts)
    → metrics.stop()
    → reporter.report()
```

**POC Validation Results:**
- ✅ PyTorch backend produces comparable energy metrics to optimum-benchmark
- ✅ Both backends correctly measure GPU energy consumption
- ✅ Configuration format compatible (Hydra-based)
- ✅ Similar performance characteristics (throughput, latency)

### 1.3 neuralwatt_cloud Integration

**Current Usage:**
```bash
# neuralwatt_cloud/run-benchmark-genai.sh
# Uses vLLM endpoint + genai-perf load generator
# Results uploaded to ClickHouse for Q-learning

./run-benchmark-genai.sh \
  --profile moderate \
  --llm llama-3.3-70b \
  --duration 300
```

**Requirements:**
- Use ai_energy_benchmarks as a library
- Support vLLM backend for production load testing
- Integrate with ClickHouse for metrics storage
- Support Q-learning experiment workflows
- Compatible with existing shell scripts

---

## 2. Backend Switching Architecture

### 2.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User/Application Layer                   │
│  (AIEnergyScore, neuralwatt_cloud, standalone CLI)          │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Backend Selection Layer (NEW)                   │
│  - Environment-based selection (BENCHMARK_BACKEND env var)  │
│  - Config-based selection (backend.engine: optimum|pytorch) │
│  - Programmatic API selection                                │
└─────────────────────────────────────────────────────────────┘
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
┌───────────────────────────┐  ┌──────────────────────────────┐
│   optimum-benchmark        │  │  ai_energy_benchmarks        │
│   (HuggingFace)            │  │  (NeuralWatt)                │
│                            │  │                              │
│   Backend: PyTorch         │  │   Backends: PyTorch, vLLM    │
│   Config: Hydra            │  │   Config: Hydra              │
│   Metrics: optimum trackers│  │   Metrics: CodeCarbon        │
└───────────────────────────┘  └──────────────────────────────┘
                │                            │
                └────────────┬───────────────┘
                             ▼
                   ┌──────────────────────┐
                   │   GPU Hardware        │
                   │   (H100, B200, etc.)  │
                   └──────────────────────┘
```

### 2.2 Backend Selection Mechanisms

#### Option A: Environment Variable (Recommended for AIEnergyScore)

```bash
# Use optimum-benchmark (default)
export BENCHMARK_BACKEND=optimum
docker run --gpus all energy_star --config-name text_generation

# Use ai_energy_benchmarks PyTorch
export BENCHMARK_BACKEND=pytorch
docker run --gpus all energy_star --config-name text_generation

# Use ai_energy_benchmarks vLLM (requires vLLM server)
export BENCHMARK_BACKEND=vllm
export VLLM_ENDPOINT=http://localhost:8000/v1
docker run --gpus all energy_star --config-name text_generation
```

#### Option B: Configuration File (Recommended for neuralwatt_cloud)

```yaml
# configs/text_generation.yaml
defaults:
  - benchmark
  - backend: pytorch  # or: optimum, vllm
  - launcher: process
  - scenario: energy_star

backend:
  engine: pytorch  # NEW: optimum|pytorch|vllm
  # engine: optimum uses optimum-benchmark
  # engine: pytorch uses ai_energy_benchmarks PyTorchBackend
  # engine: vllm uses ai_energy_benchmarks VLLMBackend

  model: openai/gpt-oss-120b
  device: cuda
  device_ids: 0
```

#### Option C: Programmatic API (Recommended for neuralwatt_cloud)

```python
# Python library usage
from ai_energy_benchmarks import BenchmarkRunner, BackendType
from ai_energy_benchmarks.config import ConfigParser

# Load config and select backend
config = ConfigParser.load_config('text_generation.yaml')
config.backend.engine = BackendType.PYTORCH  # or VLLM, OPTIMUM

# Run benchmark
runner = BenchmarkRunner(config)
results = runner.run()

# Or use factory pattern
from ai_energy_benchmarks import create_benchmark

benchmark = create_benchmark(
    backend='pytorch',  # or 'vllm', 'optimum'
    model='openai/gpt-oss-120b',
    config_path='text_generation.yaml'
)
results = benchmark.run()
```

### 2.3 Unified Result Format

Both backends must produce compatible output:

```json
{
  "config": {
    "backend_engine": "pytorch|optimum|vllm",
    "backend_type": "pytorch|vllm",
    "model": "openai/gpt-oss-120b",
    "dataset": "EnergyStarAI/text_generation",
    "num_samples": 1000
  },
  "energy": {
    "gpu_energy_wh": 125.4,
    "gpu_energy_kwh": 0.1254,
    "cpu_energy_wh": 15.2,
    "ram_energy_wh": 8.1,
    "total_energy_wh": 148.7,
    "emissions_g_co2eq": 89.3,
    "carbon_intensity_g_per_kwh": 600.0,
    "duration_seconds": 300.0
  },
  "performance": {
    "total_prompts": 1000,
    "successful_prompts": 998,
    "failed_prompts": 2,
    "total_tokens": 125000,
    "throughput_tokens_per_second": 416.7,
    "avg_latency_seconds": 0.24,
    "p50_latency_seconds": 0.22,
    "p95_latency_seconds": 0.35,
    "p99_latency_seconds": 0.48
  },
  "backend_info": {
    "backend_engine": "pytorch",
    "inference_mode": "direct",
    "gpu_model": "NVIDIA H100 80GB",
    "cuda_version": "12.9",
    "pytorch_version": "2.8.0"
  }
}
```

---

## 3. ai_energy_benchmarks Package Enhancements

### 3.1 Package as Installable Dependency

**Goal:** Make ai_energy_benchmarks installable and usable as a library

#### 3.1.1 Enhanced pyproject.toml

```toml
[project]
name = "ai_energy_benchmarks"
version = "0.2.0"  # Post-POC
description = "Modular benchmarking framework for AI energy measurements with multiple backends"
requires-python = ">=3.10"

dependencies = [
    "requests>=2.31.0",
    "datasets>=2.14.0",
    "codecarbon>=2.3.0",
    "omegaconf>=2.3.0",
    "pyyaml>=6.0",
    "pynvml>=11.5.0",
]

[project.optional-dependencies]
# Minimal install (core + vLLM client)
vllm = []

# PyTorch backend
pytorch = [
    "torch>=2.0.0",
    "transformers>=4.30.0",
]

# optimum-benchmark compatibility layer
optimum = [
    "ai_energy_benchmarks[pytorch]",
]

# neuralwatt_cloud integration
neuralwatt = [
    "clickhouse-driver>=0.2.0",
    "mlflow>=2.0.0",
]

# Full installation
all = [
    "ai_energy_benchmarks[pytorch,vllm,optimum,neuralwatt]",
]

# Development
dev = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "ruff>=0.1.0",
    "mypy>=1.5.0",
]

[project.scripts]
ai-energy-benchmark = "ai_energy_benchmarks.cli:main"

[project.urls]
Homepage = "https://github.com/neuralwatt/ai_energy_benchmarks"
Repository = "https://github.com/neuralwatt/ai_energy_benchmarks"
```

**Installation Examples:**

```bash
# Minimal install (vLLM client only)
pip install ai_energy_benchmarks

# With PyTorch backend
pip install ai_energy_benchmarks[pytorch]

# With all backends
pip install ai_energy_benchmarks[all]

# Development install
pip install -e ".[dev]"
```

#### 3.1.2 Public API Design

```python
# ai_energy_benchmarks/__init__.py

from ai_energy_benchmarks.runner import BenchmarkRunner
from ai_energy_benchmarks.config.parser import ConfigParser, BenchmarkConfig
from ai_energy_benchmarks.backends import BackendType, PyTorchBackend, VLLMBackend
from ai_energy_benchmarks.factory import create_benchmark

__version__ = "0.2.0"

__all__ = [
    "BenchmarkRunner",
    "ConfigParser",
    "BenchmarkConfig",
    "BackendType",
    "PyTorchBackend",
    "VLLMBackend",
    "create_benchmark",
]
```

#### 3.1.3 Factory Pattern for Easy Usage

```python
# ai_energy_benchmarks/factory.py

from typing import Optional, Dict, Any
from ai_energy_benchmarks.runner import BenchmarkRunner
from ai_energy_benchmarks.config.parser import ConfigParser

def create_benchmark(
    backend: str,
    model: str,
    config_path: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
    **kwargs
) -> BenchmarkRunner:
    """Factory function to create benchmark runner.

    Args:
        backend: Backend type ('pytorch', 'vllm', 'optimum')
        model: Model identifier
        config_path: Optional path to config file
        overrides: Optional config overrides
        **kwargs: Additional config parameters

    Returns:
        Configured BenchmarkRunner instance

    Example:
        >>> benchmark = create_benchmark(
        ...     backend='pytorch',
        ...     model='openai/gpt-oss-120b',
        ...     config_path='configs/text_generation.yaml'
        ... )
        >>> results = benchmark.run()
    """
    # Load base config or create default
    if config_path:
        config = ConfigParser.load_config(config_path)
    else:
        config = ConfigParser.create_default_config()

    # Apply overrides
    config.backend.engine = backend
    config.backend.model = model

    if overrides:
        config = ConfigParser.apply_overrides(config, overrides)

    # Apply kwargs
    for key, value in kwargs.items():
        setattr(config, key, value)

    return BenchmarkRunner(config)
```

### 3.2 CLI Enhancements

#### 3.2.1 Main CLI Entry Point

```python
# ai_energy_benchmarks/cli.py

import sys
import argparse
from pathlib import Path
from ai_energy_benchmarks.runner import run_benchmark_from_config
from ai_energy_benchmarks.factory import create_benchmark

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AI Energy Benchmarks - Multi-backend benchmarking framework"
    )

    subparsers = parser.add_subparsers(dest='command', help='Available commands')

    # Run benchmark
    run_parser = subparsers.add_parser('run', help='Run benchmark')
    run_parser.add_argument('config', help='Path to configuration file')
    run_parser.add_argument('--backend', choices=['pytorch', 'vllm', 'optimum'],
                           help='Override backend engine')
    run_parser.add_argument('--model', help='Override model name')
    run_parser.add_argument('--output', help='Override output directory')
    run_parser.add_argument('overrides', nargs='*',
                           help='Config overrides (key=value format)')

    # Validate config
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    validate_parser.add_argument('config', help='Path to configuration file')

    # List backends
    list_parser = subparsers.add_parser('list-backends', help='List available backends')

    args = parser.parse_args()

    if args.command == 'run':
        run_benchmark_command(args)
    elif args.command == 'validate':
        validate_config_command(args)
    elif args.command == 'list-backends':
        list_backends_command()
    else:
        parser.print_help()
        sys.exit(1)

def run_benchmark_command(args):
    """Run benchmark from command line."""
    # Parse overrides
    overrides = {}
    for override in args.overrides:
        key, value = override.split('=', 1)
        overrides[key] = value

    # Add CLI args to overrides
    if args.backend:
        overrides['backend.engine'] = args.backend
    if args.model:
        overrides['backend.model'] = args.model
    if args.output:
        overrides['output_dir'] = args.output

    # Run benchmark
    try:
        results = run_benchmark_from_config(args.config, overrides)
        print(f"\n✓ Benchmark completed successfully")
        print(f"  Energy: {results['energy']['total_energy_wh']:.2f} Wh")
        print(f"  Throughput: {results['performance']['throughput_tokens_per_second']:.1f} tok/s")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Benchmark failed: {e}")
        sys.exit(1)

def validate_config_command(args):
    """Validate configuration file."""
    from ai_energy_benchmarks.config.parser import ConfigParser

    try:
        config = ConfigParser.load_config(args.config)
        ConfigParser.validate_config(config)
        print(f"✓ Configuration valid: {args.config}")
        sys.exit(0)
    except Exception as e:
        print(f"✗ Configuration invalid: {e}")
        sys.exit(1)

def list_backends_command():
    """List available backends."""
    from ai_energy_benchmarks.backends import get_available_backends

    backends = get_available_backends()
    print("Available backends:")
    for backend, available in backends.items():
        status = "✓" if available else "✗ (missing dependencies)"
        print(f"  {backend}: {status}")

if __name__ == '__main__':
    main()
```

**CLI Usage Examples:**

```bash
# Run with config file
ai-energy-benchmark run configs/text_generation.yaml

# Override backend
ai-energy-benchmark run configs/text_generation.yaml --backend pytorch

# Override multiple settings
ai-energy-benchmark run configs/text_generation.yaml \
  --backend vllm \
  --model openai/gpt-oss-120b \
  backend.endpoint=http://localhost:8000/v1

# Validate config
ai-energy-benchmark validate configs/text_generation.yaml

# List available backends
ai-energy-benchmark list-backends
```

### 3.3 Backend Abstraction Enhancements

#### 3.3.1 Backend Type Enum

```python
# ai_energy_benchmarks/backends/__init__.py

from enum import Enum
from typing import Dict

class BackendType(str, Enum):
    """Supported backend types."""
    PYTORCH = "pytorch"
    VLLM = "vllm"
    OPTIMUM = "optimum"  # Compatibility layer

def get_available_backends() -> Dict[str, bool]:
    """Check which backends are available.

    Returns:
        Dict mapping backend name to availability
    """
    backends = {}

    # Check PyTorch
    try:
        import torch
        import transformers
        backends['pytorch'] = True
    except ImportError:
        backends['pytorch'] = False

    # vLLM is always available (HTTP client only)
    backends['vllm'] = True

    # optimum is available if pytorch is
    backends['optimum'] = backends['pytorch']

    return backends

__all__ = [
    "BackendType",
    "get_available_backends",
    "PyTorchBackend",
    "VLLMBackend",
]
```

#### 3.3.2 Backend Factory

```python
# ai_energy_benchmarks/backends/factory.py

from typing import Dict, Any
from ai_energy_benchmarks.backends import BackendType
from ai_energy_benchmarks.backends.base import Backend
from ai_energy_benchmarks.backends.pytorch import PyTorchBackend
from ai_energy_benchmarks.backends.vllm import VLLMBackend

def create_backend(backend_type: str, config: Dict[str, Any]) -> Backend:
    """Create backend instance from type and config.

    Args:
        backend_type: Backend type string
        config: Backend configuration

    Returns:
        Backend instance

    Raises:
        ValueError: If backend type unknown or dependencies missing
    """
    backend_enum = BackendType(backend_type)

    if backend_enum == BackendType.PYTORCH:
        return PyTorchBackend(
            model=config.get('model'),
            device=config.get('device', 'cuda'),
            device_ids=config.get('device_ids', [0]),
            torch_dtype=config.get('torch_dtype', 'auto'),
            device_map=config.get('device_map', 'auto')
        )

    elif backend_enum == BackendType.VLLM:
        return VLLMBackend(
            endpoint=config.get('endpoint'),
            model=config.get('model'),
            timeout=config.get('timeout', 300)
        )

    elif backend_enum == BackendType.OPTIMUM:
        # Compatibility layer - uses PyTorch backend
        # but with optimum-benchmark-compatible behavior
        return PyTorchBackend(
            model=config.get('model'),
            device=config.get('device', 'cuda'),
            device_ids=config.get('device_ids', [0]),
            optimum_compat_mode=True  # Special flag
        )

    else:
        raise ValueError(f"Unknown backend type: {backend_type}")
```

---

## 4. Integration Strategies

### 4.1 AIEnergyScore Integration (Drop-in Replacement)

**Goal:** Enable AIEnergyScore to use ai_energy_benchmarks with minimal changes

#### 4.1.1 Modified Dockerfile

```dockerfile
# AIEnergyScore/Dockerfile (MODIFIED)

FROM pytorch/pytorch:2.8.0-cuda12.9-cudnn9-runtime

ENV PATH=/opt/conda/bin:$PATH

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git && \
        rm -rf /var/lib/apt/lists/*

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Install optimum-benchmark (default)
RUN git clone https://github.com/huggingface/optimum-benchmark.git /optimum-benchmark && \
    cd /optimum-benchmark && \
    git checkout reasoning_test && \
    pip install -e .

# Install ai_energy_benchmarks (optional)
RUN pip install ai_energy_benchmarks[pytorch]

COPY ./check_h100.py /check_h100.py
COPY ./entrypoint.sh /entrypoint.sh
COPY ./summarize_gpu_wh.py /summarize_gpu_wh.py
RUN chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]
```

#### 4.1.2 Modified entrypoint.sh (Backend Selection)

```bash
#!/bin/bash
# AIEnergyScore/entrypoint.sh (MODIFIED)

set -e

RESULTS_DIR="/results"
BENCHMARK_BACKEND="${BENCHMARK_BACKEND:-optimum}"  # Default to optimum

python /check_h100.py
if [[ $? = 0 ]]; then
    mkdir -p "${RESULTS_DIR}"

    if [[ "${BENCHMARK_BACKEND}" == "optimum" ]]; then
        echo "Using optimum-benchmark (HuggingFace)"
        optimum-benchmark --config-dir /optimum-benchmark/energy_star/ "$@" \
            hydra.run.dir="${RESULTS_DIR}"

    elif [[ "${BENCHMARK_BACKEND}" == "pytorch" ]]; then
        echo "Using ai_energy_benchmarks (PyTorch backend)"
        ai-energy-benchmark run "$@" \
            --backend pytorch \
            --output "${RESULTS_DIR}"

    elif [[ "${BENCHMARK_BACKEND}" == "vllm" ]]; then
        echo "Using ai_energy_benchmarks (vLLM backend)"
        ai-energy-benchmark run "$@" \
            --backend vllm \
            --output "${RESULTS_DIR}"

    else
        echo "Error: Unknown BENCHMARK_BACKEND=${BENCHMARK_BACKEND}"
        echo "Valid options: optimum, pytorch, vllm"
        exit 1
    fi

    # Post-run summarizer works with both backends
    python /summarize_gpu_wh.py "${RESULTS_DIR}"
fi
```

#### 4.1.3 Usage Examples

```bash
# Default: optimum-benchmark
docker run --gpus all energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b

# Use ai_energy_benchmarks PyTorch
docker run --gpus all \
  -e BENCHMARK_BACKEND=pytorch \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b

# Use ai_energy_benchmarks vLLM (requires vLLM server)
docker run --gpus all \
  -e BENCHMARK_BACKEND=vllm \
  -e VLLM_ENDPOINT=http://host.docker.internal:8000/v1 \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

### 4.2 neuralwatt_cloud Integration (Library Usage)

**Goal:** Use ai_energy_benchmarks as a Python library from neuralwatt_cloud

#### 4.2.1 Installation in neuralwatt_cloud

```bash
# neuralwatt_cloud/requirements.txt (ADD)
ai_energy_benchmarks[all]>=0.2.0
```

#### 4.2.2 Python Integration Module

```python
# neuralwatt_cloud/benchmarking/ai_energy_integration.py (NEW)

"""Integration with ai_energy_benchmarks for benchmark execution."""

from typing import Dict, Any, Optional
from ai_energy_benchmarks import create_benchmark
from ai_energy_benchmarks.config import ConfigParser
import logging

logger = logging.getLogger(__name__)

class AIEnergyBenchmarkRunner:
    """Wrapper for ai_energy_benchmarks in neuralwatt_cloud context."""

    def __init__(
        self,
        backend: str = 'vllm',
        model: str = None,
        endpoint: str = 'http://localhost:8000/v1',
        config_path: Optional[str] = None
    ):
        """Initialize benchmark runner.

        Args:
            backend: Backend type ('pytorch', 'vllm')
            model: Model identifier
            endpoint: vLLM endpoint (for vllm backend)
            config_path: Optional config file path
        """
        self.backend = backend
        self.model = model
        self.endpoint = endpoint
        self.config_path = config_path

    def run_benchmark(
        self,
        profile: str = 'moderate',
        duration: int = 300,
        dataset: str = 'EnergyStarAI/text_generation',
        num_samples: int = 1000,
        **kwargs
    ) -> Dict[str, Any]:
        """Run benchmark with specified parameters.

        Args:
            profile: Load profile (light, moderate, heavy, stress)
            duration: Test duration in seconds
            dataset: Dataset identifier
            num_samples: Number of samples to process
            **kwargs: Additional config overrides

        Returns:
            Benchmark results dictionary
        """
        logger.info(f"Running benchmark: backend={self.backend}, model={self.model}")

        # Create benchmark config
        config_overrides = {
            'scenario.dataset_name': dataset,
            'scenario.num_samples': num_samples,
            'scenario.duration': duration,
            'scenario.profile': profile,
        }

        if self.backend == 'vllm':
            config_overrides['backend.endpoint'] = self.endpoint

        config_overrides.update(kwargs)

        # Create and run benchmark
        benchmark = create_benchmark(
            backend=self.backend,
            model=self.model,
            config_path=self.config_path,
            overrides=config_overrides
        )

        results = benchmark.run()

        logger.info(f"Benchmark completed: {results['summary']['total_prompts']} prompts")
        logger.info(f"Energy: {results['energy']['total_energy_wh']:.2f} Wh")

        return results

    def upload_to_clickhouse(self, results: Dict[str, Any]):
        """Upload results to ClickHouse (neuralwatt_cloud integration)."""
        from neuralwatt_cloud.database.clickhouse import upload_benchmark_results

        # Transform ai_energy_benchmarks format to neuralwatt_cloud format
        clickhouse_data = self._transform_results(results)
        upload_benchmark_results(clickhouse_data)

    def _transform_results(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Transform results to neuralwatt_cloud format."""
        # Map ai_energy_benchmarks format to neuralwatt_cloud format
        return {
            'run_id': results.get('run_id'),
            'timestamp': results.get('timestamp'),
            'model': results['config']['model'],
            'backend': results['config']['backend_engine'],
            'total_energy_wh': results['energy']['total_energy_wh'],
            'gpu_energy_wh': results['energy']['gpu_energy_wh'],
            'throughput': results['performance']['throughput_tokens_per_second'],
            # ... more mappings
        }
```

#### 4.2.3 Modified run-benchmark-genai.sh

```bash
#!/bin/bash
# neuralwatt_cloud/run-benchmark-genai.sh (MODIFIED)

# ... existing setup ...

# NEW: Option to use ai_energy_benchmarks
USE_AI_ENERGY_BENCHMARKS="${USE_AI_ENERGY_BENCHMARKS:-false}"

if [[ "${USE_AI_ENERGY_BENCHMARKS}" == "true" ]]; then
    echo "Using ai_energy_benchmarks library"

    python -c "
from neuralwatt_cloud.benchmarking.ai_energy_integration import AIEnergyBenchmarkRunner

runner = AIEnergyBenchmarkRunner(
    backend='${BACKEND}',
    model='${AI_MODEL}',
    endpoint='${ENDPOINT}'
)

results = runner.run_benchmark(
    profile='${PROFILE}',
    duration=${DURATION},
    num_samples=${NUM_SAMPLES}
)

# Upload to ClickHouse
runner.upload_to_clickhouse(results)

print('Benchmark complete')
"
else
    # Existing genai_perf implementation
    echo "Using existing genai_perf implementation"
    # ... existing code ...
fi
```

#### 4.2.4 Usage Examples

```bash
# Use ai_energy_benchmarks with vLLM backend
USE_AI_ENERGY_BENCHMARKS=true \
BACKEND=vllm \
./run-benchmark-genai.sh \
  --profile moderate \
  --llm llama-3.3-70b \
  --duration 300

# Use ai_energy_benchmarks with PyTorch backend (for baselines)
USE_AI_ENERGY_BENCHMARKS=true \
BACKEND=pytorch \
./run-benchmark-genai.sh \
  --profile moderate \
  --llm llama-3.3-70b \
  --duration 300
```

### 4.3 Standalone Usage

**Direct CLI usage (no integration required):**

```bash
# PyTorch backend
ai-energy-benchmark run configs/pytorch_validation.yaml

# vLLM backend
ai-energy-benchmark run configs/vllm_benchmark.yaml \
  backend.endpoint=http://localhost:8000/v1

# With overrides
ai-energy-benchmark run configs/text_generation.yaml \
  --backend vllm \
  --model openai/gpt-oss-120b \
  scenario.num_samples=500
```

---

## 5. Configuration Management

### 5.1 Unified Configuration Format

Both backends use Hydra/OmegaConf for configuration compatibility.

#### 5.1.1 Base Configuration Structure

```yaml
# configs/benchmark/default.yaml
defaults:
  - backend: pytorch  # or vllm, optimum
  - launcher: process
  - scenario: energy_star
  - _self_

name: benchmark_default
output_dir: ./results

# Launcher config
launcher:
  device_isolation: false
  device_isolation_action: warn

# Metrics config
metrics:
  enabled: true
  type: codecarbon
  project_name: "ai_energy_benchmark"
  country_iso_code: "USA"
  region: "california"
  output_dir: "./emissions"

# Reporter config
reporter:
  type: csv
  output_file: "./results/benchmark_{timestamp}.csv"
```

#### 5.1.2 Backend-Specific Configs

```yaml
# configs/backend/pytorch.yaml
type: pytorch
engine: pytorch  # NEW: distinguishes from optimum
device: cuda
device_ids: [0]
model: openai/gpt-oss-120b
task: text-generation
torch_dtype: auto
device_map: auto
optimum_compat_mode: false
```

```yaml
# configs/backend/vllm.yaml
type: vllm
engine: vllm
device: cuda
device_ids: [0]
model: openai/gpt-oss-120b
task: text-generation
endpoint: "http://localhost:8000/v1"
timeout: 300
```

```yaml
# configs/backend/optimum.yaml (COMPATIBILITY)
type: pytorch
engine: optimum  # Special flag
device: cuda
device_ids: [0]
model: openai/gpt-oss-120b
task: text-generation
torch_dtype: auto
device_map: auto
optimum_compat_mode: true  # Enable optimum-benchmark compatibility
```

#### 5.1.3 Complete Example Config

```yaml
# configs/text_generation_multibackend.yaml
defaults:
  - benchmark: default
  - backend: pytorch  # Can override with: vllm, optimum
  - launcher: process
  - scenario: energy_star
  - _self_

name: text_generation_benchmark

backend:
  # Override backend engine
  # engine: pytorch|vllm|optimum
  model: openai/gpt-oss-120b
  device: cuda
  device_ids: [0]

  # PyTorch-specific (when engine=pytorch or optimum)
  torch_dtype: auto
  device_map: auto

  # vLLM-specific (when engine=vllm)
  endpoint: "http://localhost:8000/v1"
  timeout: 300

scenario:
  dataset_name: EnergyStarAI/text_generation
  text_column_name: text
  num_samples: 1000
  truncation: true
  reasoning: false

  input_shapes:
    batch_size: 1

  generate_kwargs:
    max_new_tokens: 100
    min_new_tokens: 50

metrics:
  enabled: true
  type: codecarbon
  project_name: "text_generation_${backend.engine}"
  gpu_ids: ${backend.device_ids}

output_dir: ./results/${backend.engine}_${now:%Y%m%d_%H%M%S}
```

### 5.2 Configuration Migration Strategy

#### 5.2.1 Migration Tool

```python
# ai_energy_benchmarks/utils/config_migration.py

def migrate_optimum_config_to_multibackend(config_path: str) -> Dict[str, Any]:
    """Migrate optimum-benchmark config to multi-backend format.

    Args:
        config_path: Path to optimum-benchmark config

    Returns:
        Multi-backend compatible config
    """
    from omegaconf import OmegaConf

    config = OmegaConf.load(config_path)

    # Add engine field
    if 'backend' in config:
        if not hasattr(config.backend, 'engine'):
            config.backend.engine = 'optimum'

    return OmegaConf.to_container(config)
```

#### 5.2.2 Backward Compatibility

Existing optimum-benchmark configs work without modification:

```yaml
# Existing AIEnergyScore config (NO CHANGES NEEDED)
defaults:
  - benchmark
  - backend: pytorch
  - launcher: process
  - scenario: energy_star

name: text_generation_validation

backend:
  model: openai/gpt-oss-120b
  # ... existing config ...

# Backend engine automatically determined:
# - If using optimum-benchmark: engine=optimum
# - If using ai_energy_benchmarks: engine=pytorch (default)
```

---

## 6. Implementation Roadmap

### Phase 1: Package ai_energy_benchmarks for Distribution (1 week)

**Objectives:**
- Make ai_energy_benchmarks installable via pip
- Publish to PyPI or private package repository
- Enable library usage in other projects

**Tasks:**
1. **Enhanced pyproject.toml** ✓
   - Optional dependencies structure
   - Proper versioning
   - Entry points

2. **Public API Design** ✓
   - `__init__.py` exports
   - Factory functions
   - Type hints throughout

3. **CLI Enhancements** ✓
   - Improved `ai-energy-benchmark` command
   - Subcommands: run, validate, list-backends
   - Better error messages

4. **Documentation**
   - Installation guide
   - API reference
   - Usage examples

5. **Testing**
   - Unit tests for public API
   - Integration tests
   - CI/CD for package building

6. **Release**
   - Version 0.2.0
   - Package distribution
   - Release notes

**Deliverables:**
- [ ] Installable package (pip install ai_energy_benchmarks)
- [ ] Public API documented
- [ ] CLI functional
- [ ] Tests passing
- [ ] Package published

### Phase 2: Backend Selection & Adapter Layer (1 week)

**Objectives:**
- Implement backend selection mechanism
- Create compatibility layer for optimum-benchmark
- Enable seamless switching

**Tasks:**
1. **Backend Type System**
   - BackendType enum
   - Backend detection
   - Dependency checking

2. **Backend Factory**
   - Dynamic backend creation
   - Configuration mapping
   - Error handling

3. **Configuration Enhancements**
   - `engine` field support
   - Environment variable support
   - Override mechanisms

4. **Compatibility Layer** (Optional)
   - optimum-benchmark wrapper
   - Result format translation
   - API compatibility

5. **Testing**
   - Backend switching tests
   - Config validation tests
   - Compatibility tests

**Deliverables:**
- [ ] Backend selection working
- [ ] Configuration enhanced
- [ ] Tests passing
- [ ] Documentation updated

### Phase 3: Integration with AIEnergyScore (1 week)

**Objectives:**
- Integrate ai_energy_benchmarks into AIEnergyScore
- Maintain backward compatibility
- Enable backend switching

**Tasks:**
1. **Dockerfile Modifications**
   - Install ai_energy_benchmarks
   - Preserve optimum-benchmark
   - Environment setup

2. **Entrypoint Script**
   - Backend selection logic
   - Conditional execution
   - Error handling

3. **Result Summarization**
   - Update summarize_gpu_wh.py
   - Support both result formats
   - Maintain output format

4. **Configuration**
   - Test existing configs
   - Create multi-backend examples
   - Documentation

5. **Testing**
   - Docker build tests
   - End-to-end benchmarks
   - Result validation
   - Compare optimum vs ai_energy_benchmarks outputs

**Deliverables:**
- [ ] AIEnergyScore Docker image updated
- [ ] Backend switching functional
- [ ] Tests passing
- [ ] Documentation complete

### Phase 4: Integration with neuralwatt_cloud (1 week)

**Objectives:**
- Use ai_energy_benchmarks as library in neuralwatt_cloud
- Support vLLM backend for production workloads
- Integrate with ClickHouse

**Tasks:**
1. **Integration Module**
   - AIEnergyBenchmarkRunner class
   - Result transformation
   - ClickHouse upload

2. **Shell Script Updates**
   - Modify run-benchmark-genai.sh
   - Add USE_AI_ENERGY_BENCHMARKS flag
   - Backward compatibility

3. **Configuration**
   - neuralwatt_cloud config format
   - Model registry integration
   - Environment variables

4. **Testing**
   - Integration tests
   - End-to-end workflows
   - ClickHouse data validation
   - Q-learning compatibility

**Deliverables:**
- [ ] neuralwatt_cloud integration working
- [ ] ClickHouse uploads functional
- [ ] Tests passing
- [ ] Documentation complete

### Phase 5: Documentation & Polish (3-5 days)

**Objectives:**
- Comprehensive documentation
- Examples for all use cases
- Migration guides

**Tasks:**
1. **User Documentation**
   - Getting started guide
   - Backend comparison guide
   - Configuration reference
   - Troubleshooting

2. **Integration Guides**
   - AIEnergyScore integration
   - neuralwatt_cloud integration
   - Standalone usage

3. **Examples**
   - Example configs for each backend
   - Sample workflows
   - Common patterns

4. **Migration Guide**
   - optimum-benchmark → ai_energy_benchmarks
   - Configuration migration
   - Result format changes

**Deliverables:**
- [ ] Complete documentation
- [ ] Example repository
- [ ] Migration guide
- [ ] Troubleshooting guide

### Phase 6: Validation & Release (3-5 days)

**Objectives:**
- Comprehensive testing
- Result validation
- Production release

**Tasks:**
1. **Validation Testing**
   - Compare all backends
   - Energy metric accuracy
   - Performance benchmarks
   - Result consistency

2. **Production Testing**
   - Test on target hardware (H100, B200)
   - Long-duration runs
   - Stress testing
   - Error handling

3. **Release Preparation**
   - Version 1.0.0 planning
   - Release notes
   - Announcement
   - Support plan

**Deliverables:**
- [ ] All tests passing
- [ ] Production validation complete
- [ ] Version 1.0.0 released
- [ ] Documentation live

---

## 7. Testing & Validation Strategy

### 7.1 Unit Tests

**Coverage Target:** >80%

**Test Areas:**
- Backend factory
- Configuration parser
- Backend selection logic
- Result format conversion
- API functions

**Example Tests:**
```python
# tests/test_backend_selection.py

def test_backend_type_enum():
    assert BackendType.PYTORCH == "pytorch"
    assert BackendType.VLLM == "vllm"
    assert BackendType.OPTIMUM == "optimum"

def test_backend_factory_pytorch():
    config = {'model': 'test/model', 'device': 'cuda'}
    backend = create_backend('pytorch', config)
    assert isinstance(backend, PyTorchBackend)

def test_backend_factory_vllm():
    config = {'model': 'test/model', 'endpoint': 'http://localhost:8000/v1'}
    backend = create_backend('vllm', config)
    assert isinstance(backend, VLLMBackend)

def test_config_backend_selection():
    config = ConfigParser.load_config('test_config.yaml')
    config.backend.engine = 'pytorch'
    assert config.backend.engine == BackendType.PYTORCH
```

### 7.2 Integration Tests

**Test Scenarios:**

1. **Backend Switching**
   - Same config, different backends
   - Verify both produce results
   - Compare result formats

2. **AIEnergyScore Integration**
   - Docker build
   - Backend selection via env var
   - Result extraction

3. **neuralwatt_cloud Integration**
   - Library import
   - Benchmark execution
   - ClickHouse upload

**Example Test:**
```python
# tests/integration/test_backend_comparison.py

def test_pytorch_vs_optimum_compatibility():
    """Compare PyTorch backend vs optimum-benchmark compatibility mode."""
    config = load_test_config('text_generation.yaml')

    # Run with pytorch backend
    config.backend.engine = 'pytorch'
    runner1 = BenchmarkRunner(config)
    results1 = runner1.run()

    # Run with optimum compatibility
    config.backend.engine = 'optimum'
    runner2 = BenchmarkRunner(config)
    results2 = runner2.run()

    # Results should be comparable
    assert_results_comparable(results1, results2, tolerance=0.05)
```

### 7.3 End-to-End Validation

**Critical Path Tests:**

1. **AIEnergyScore Workflow**
   ```bash
   # Test optimum-benchmark (baseline)
   docker run --gpus all energy_star \
     --config-name text_generation_validation

   # Test ai_energy_benchmarks PyTorch
   docker run --gpus all -e BENCHMARK_BACKEND=pytorch energy_star \
     --config-name text_generation_validation

   # Compare GPU_ENERGY_WH outputs
   ```

2. **neuralwatt_cloud Workflow**
   ```bash
   # Test with ai_energy_benchmarks
   USE_AI_ENERGY_BENCHMARKS=true \
   BACKEND=vllm \
   ./run-benchmark-genai.sh --profile moderate --duration 60

   # Verify ClickHouse upload
   # Verify Q-learning integration
   ```

3. **Standalone Workflow**
   ```bash
   # Direct CLI usage
   ai-energy-benchmark run configs/pytorch_validation.yaml
   ai-energy-benchmark run configs/vllm_benchmark.yaml
   ```

### 7.4 Result Validation

**Comparison Metrics:**

| Metric | Tolerance | Notes |
|--------|-----------|-------|
| GPU Energy (Wh) | ±5% | Primary metric |
| Total Energy (Wh) | ±5% | Includes CPU/RAM |
| Throughput (tok/s) | ±10% | May vary with load |
| Latency (s) | ±10% | May vary with load |
| CO₂ emissions | ±5% | Based on energy |

**Validation Script:**
```python
# tests/validation/compare_backends.py

def validate_backend_results(optimum_results, pytorch_results):
    """Validate that PyTorch backend results match optimum-benchmark."""

    # Energy comparison
    gpu_energy_diff = abs(
        optimum_results['energy']['gpu_energy_wh'] -
        pytorch_results['energy']['gpu_energy_wh']
    ) / optimum_results['energy']['gpu_energy_wh']

    assert gpu_energy_diff < 0.05, \
        f"GPU energy difference too large: {gpu_energy_diff:.2%}"

    # Throughput comparison
    throughput_diff = abs(
        optimum_results['performance']['throughput_tokens_per_second'] -
        pytorch_results['performance']['throughput_tokens_per_second']
    ) / optimum_results['performance']['throughput_tokens_per_second']

    assert throughput_diff < 0.10, \
        f"Throughput difference too large: {throughput_diff:.2%}"

    print("✓ Backend results validated - within acceptable tolerance")
```

### 7.5 Hardware Validation Matrix

**Test on Multiple GPUs:**

| GPU Model | optimum-benchmark | ai_energy_benchmarks PyTorch | ai_energy_benchmarks vLLM |
|-----------|-------------------|------------------------------|---------------------------|
| H100 80GB | ✓ Baseline | ✓ Compare | ✓ Test |
| B200 | ✓ Baseline | ✓ Compare | ✓ Test |
| RTX Pro 6000 | ✓ Test | ✓ Validated (POC) | ✓ Validated (POC) |

---

## 8. Decision Points & Recommendations

### 8.1 Backend Selection Strategy

**Recommendation: Option A - Environment Variable** (for AIEnergyScore)

**Rationale:**
- ✅ Minimal code changes to AIEnergyScore
- ✅ Docker-friendly (docker run -e BENCHMARK_BACKEND=pytorch)
- ✅ Clear separation of concerns
- ✅ Easy to switch during runtime
- ✅ Backward compatible (defaults to optimum)

**Implementation:**
- Modify entrypoint.sh to check `$BENCHMARK_BACKEND`
- Default to `optimum` if not set
- Support `pytorch`, `vllm`, `optimum` values

### 8.2 neuralwatt_cloud Integration

**Recommendation: Option C - Programmatic API** (for neuralwatt_cloud)

**Rationale:**
- ✅ Clean Python API
- ✅ Easy to integrate with existing code
- ✅ Type-safe
- ✅ Testable
- ✅ Flexible for future extensions

**Implementation:**
- Create `AIEnergyBenchmarkRunner` wrapper class
- Import as library: `from ai_energy_benchmarks import create_benchmark`
- Support both PyTorch and vLLM backends

### 8.3 Package Distribution

**Recommendation: PyPI Public Package**

**Rationale:**
- ✅ Easy installation: `pip install ai_energy_benchmarks`
- ✅ Versioning and dependency management
- ✅ Community visibility
- ✅ Standard Python packaging

**Alternative:** Private package repository (if needed for proprietary code)

### 8.4 Backward Compatibility

**Recommendation: Full Backward Compatibility**

**Rationale:**
- ✅ No breaking changes to AIEnergyScore
- ✅ Existing configs work without modification
- ✅ optimum-benchmark remains default
- ✅ Gradual migration path

**Implementation:**
- Default to optimum-benchmark if `BENCHMARK_BACKEND` not set
- Accept all existing config formats
- Result format compatible with existing tools

### 8.5 optimum-benchmark Dependency

**Recommendation: Keep optimum-benchmark as Optional Dependency**

**Rationale:**
- ✅ AIEnergyScore continues to use it by default
- ✅ Proven and validated by HuggingFace
- ✅ No migration pressure
- ✅ ai_energy_benchmarks is additive, not replacement

**Implementation:**
- AIEnergyScore Docker includes both packages
- Users can switch via environment variable
- Both backends maintained and tested

### 8.6 Configuration Format

**Recommendation: Maintain Hydra/OmegaConf Compatibility**

**Rationale:**
- ✅ Already used by AIEnergyScore
- ✅ Powerful composition system
- ✅ Industry standard
- ✅ No learning curve

**Implementation:**
- Use OmegaConf for all config parsing
- Maintain compatible structure
- Add `engine` field for backend selection

### 8.7 Result Format

**Recommendation: Unified Result Format with Backward Compatibility**

**Rationale:**
- ✅ Consistent across backends
- ✅ Easy to parse
- ✅ Compatible with existing tools (summarize_gpu_wh.py)

**Implementation:**
- Define common result schema
- Transform backend-specific results to common format
- Ensure GPU_ENERGY_WH extraction works for both

---

## 9. Risk Assessment

### 9.1 Technical Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| PyTorch backend results diverge from optimum-benchmark | Low | High | POC validated compatibility; continuous testing |
| CodeCarbon energy measurements differ from optimum trackers | Low | Medium | Both use similar underlying APIs; validate against hardware meters |
| Configuration incompatibility | Low | Medium | Maintain Hydra format; extensive testing |
| Package dependency conflicts | Medium | Medium | Optional dependencies; version pinning |
| Performance overhead | Low | Low | Minimal abstraction; benchmark both |
| vLLM API changes | Low | Medium | Pin vLLM version; compatibility matrix |

### 9.2 Integration Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| AIEnergyScore Docker build issues | Low | Medium | Test Docker build in CI; multiple GPU types |
| neuralwatt_cloud ClickHouse incompatibility | Low | Medium | Result transformation layer; thorough testing |
| Breaking existing workflows | Low | High | Full backward compatibility; default to optimum |
| Documentation gaps | Medium | Medium | Comprehensive docs; examples; migration guide |

### 9.3 Operational Risks

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Adoption resistance | Medium | Medium | Clear benefits; gradual migration; no forced changes |
| Maintenance burden | Medium | Medium | Modular design; automated testing; clear ownership |
| Version fragmentation | Low | Medium | Semantic versioning; compatibility policy |
| Support overhead | Medium | Low | Good documentation; examples; troubleshooting guide |

---

## 10. Appendices

### Appendix A: Key Metrics for Success

**Phase 1 Success (Package Distribution):**
- [ ] Package installable via pip
- [ ] All optional dependencies working
- [ ] CLI functional
- [ ] Tests passing (>80% coverage)
- [ ] Documentation complete

**Phase 2 Success (Backend Selection):**
- [ ] Backend switching working
- [ ] Configuration enhanced
- [ ] All backends selectable
- [ ] Tests passing

**Phase 3 Success (AIEnergyScore Integration):**
- [ ] Docker build successful
- [ ] Backend switching via env var working
- [ ] optimum-benchmark still works (default)
- [ ] PyTorch backend produces comparable results (±5%)
- [ ] GPU_ENERGY_WH extraction working for both backends

**Phase 4 Success (neuralwatt_cloud Integration):**
- [ ] Library import working
- [ ] Benchmark execution successful
- [ ] ClickHouse uploads functional
- [ ] Q-learning workflows compatible

**Overall Success Criteria:**
- [ ] All three integration modes working
- [ ] Result validation passing (±5% tolerance)
- [ ] No breaking changes to existing systems
- [ ] Comprehensive documentation
- [ ] Production-ready release (v1.0.0)

### Appendix B: Configuration Examples

#### B.1 AIEnergyScore (optimum-benchmark default)

```bash
# Current usage - no changes needed
docker run --gpus all energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

#### B.2 AIEnergyScore (ai_energy_benchmarks PyTorch)

```bash
# New usage - add environment variable
docker run --gpus all \
  -e BENCHMARK_BACKEND=pytorch \
  energy_star \
  --config-name text_generation \
  backend.model=openai/gpt-oss-120b
```

#### B.3 neuralwatt_cloud (Library Usage)

```python
from ai_energy_benchmarks import create_benchmark

benchmark = create_benchmark(
    backend='vllm',
    model='openai/gpt-oss-120b',
    config_path='configs/text_generation.yaml',
    overrides={
        'backend.endpoint': 'http://localhost:8000/v1',
        'scenario.profile': 'moderate',
        'scenario.duration': 300
    }
)

results = benchmark.run()
```

#### B.4 Standalone CLI

```bash
# PyTorch backend
ai-energy-benchmark run configs/pytorch_validation.yaml

# vLLM backend with overrides
ai-energy-benchmark run configs/text_generation.yaml \
  --backend vllm \
  --model openai/gpt-oss-120b \
  backend.endpoint=http://localhost:8000/v1
```

### Appendix C: Result Format Comparison

#### C.1 optimum-benchmark Output

```json
{
  "benchmark_report": {
    "model": "openai/gpt-oss-120b",
    "energy": {
      "gpu_energy": 125.4,
      "cpu_energy": 15.2,
      "total_energy": 140.6
    },
    "performance": {
      "throughput": 420.5
    }
  }
}
```

#### C.2 ai_energy_benchmarks Output

```json
{
  "config": {
    "backend_engine": "pytorch",
    "model": "openai/gpt-oss-120b"
  },
  "energy": {
    "gpu_energy_wh": 125.4,
    "cpu_energy_wh": 15.2,
    "total_energy_wh": 148.7,
    "emissions_g_co2eq": 89.3
  },
  "performance": {
    "throughput_tokens_per_second": 416.7
  }
}
```

#### C.3 Unified Format (Target)

```json
{
  "config": {
    "backend_engine": "pytorch|optimum|vllm",
    "model": "openai/gpt-oss-120b"
  },
  "energy": {
    "gpu_energy_wh": 125.4,
    "cpu_energy_wh": 15.2,
    "ram_energy_wh": 8.1,
    "total_energy_wh": 148.7,
    "emissions_g_co2eq": 89.3
  },
  "performance": {
    "throughput_tokens_per_second": 416.7,
    "avg_latency_seconds": 0.24
  }
}
```

### Appendix D: Glossary

- **Backend**: Inference engine implementation (PyTorch, vLLM, optimum-benchmark)
- **Backend Engine**: The specific backend type selected for execution
- **optimum-benchmark**: HuggingFace's benchmarking framework (AIEnergyScore default)
- **ai_energy_benchmarks**: NeuralWatt's multi-backend benchmarking framework
- **Hydra**: Configuration management framework used by both systems
- **Drop-in Replacement**: Can substitute one component for another without code changes
- **Backward Compatibility**: New version works with old configurations/workflows
- **POC**: Proof of Concept - initial validation phase

### Appendix E: Contact & Support

**Development Team:**
- ai_energy_benchmarks: NeuralWatt team
- AIEnergyScore: HuggingFace team
- neuralwatt_cloud: NeuralWatt team

**Support Channels:**
- GitHub Issues: [ai_energy_benchmarks issues](https://github.com/neuralwatt/ai_energy_benchmarks/issues)
- Documentation: [ai_energy_benchmarks docs](https://github.com/neuralwatt/ai_energy_benchmarks/tree/main/docs)
- Email: support@neuralwatt.com

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.0 | 2025-10-07 | Claude Code | Initial design document |

**Review Status:** Design Document
**Next Review:** After stakeholder feedback

**Approvals Required:**
- [ ] AIEnergyScore Integration Lead
- [ ] neuralwatt_cloud Technical Lead
- [ ] ai_energy_benchmarks Maintainer

---

## Next Steps

1. **Review this document** with stakeholders from all three projects
2. **Validate approach** with key decision makers
3. **Prioritize phases** based on business needs
4. **Assign ownership** for each integration point
5. **Begin Phase 1** - Package ai_energy_benchmarks for distribution

**Recommended Priority:**
1. Phase 1 (Package) - Foundation for everything else
2. Phase 2 (Backend Selection) - Core switching mechanism
3. Phase 4 (neuralwatt_cloud) - Immediate business value
4. Phase 3 (AIEnergyScore) - Additive, not critical path
5. Phases 5-6 (Docs & Validation) - Ongoing throughout

**Timeline Estimate:** 4-6 weeks for full implementation
