#!/usr/bin/env python3
"""
Debug Logger for AI Energy Score Batch Runner.

Provides structured logging with file and console output for debugging
benchmark runs.
"""

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class DebugLogger:
    """Debug logger with file and console output."""

    def __init__(
        self,
        log_dir: str,
        model_name: str,
        reasoning_state: str,
        console_level: int = logging.INFO,
        file_level: int = logging.DEBUG,
    ):
        """Initialize debug logger.

        Args:
            log_dir: Directory to store log files
            model_name: Model name for log filename
            reasoning_state: Reasoning state for log filename
            console_level: Logging level for console output
            file_level: Logging level for file output
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Create unique logger name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_model = model_name.replace("/", "_").replace(" ", "_")
        safe_reasoning = (
            reasoning_state.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )
        self.log_name = f"{safe_model}_{safe_reasoning}_{timestamp}"

        # Create logger
        self.logger = logging.getLogger(self.log_name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Remove any existing handlers
        self.logger.handlers.clear()

        # Create formatters
        file_formatter = logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
        )
        console_formatter = logging.Formatter("%(levelname)s: %(message)s")

        # File handler
        log_file = self.log_dir / f"{self.log_name}.log"
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(file_level)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)

        self.log_file_path = log_file

    def info(self, message: str) -> None:
        """Log info message."""
        self.logger.info(message)

    def debug(self, message: str) -> None:
        """Log debug message."""
        self.logger.debug(message)

    def warning(self, message: str) -> None:
        """Log warning message."""
        self.logger.warning(message)

    def error(self, message: str) -> None:
        """Log error message."""
        self.logger.error(message)

    def critical(self, message: str) -> None:
        """Log critical message."""
        self.logger.critical(message)

    def log_config(self, config_dict: dict) -> None:
        """Log configuration dictionary.

        Args:
            config_dict: Configuration dictionary to log
        """
        self.info("Configuration:")
        for key, value in config_dict.items():
            self.info(f"  {key}: {value}")

    def log_benchmark_start(
        self, model_id: str, num_prompts: int, backend: str
    ) -> None:
        """Log benchmark start.

        Args:
            model_id: Model identifier
            num_prompts: Number of prompts to process
            backend: Backend type (vllm, pytorch)
        """
        self.info("=" * 80)
        self.info(f"Starting benchmark for {model_id}")
        self.info(f"Backend: {backend}")
        self.info(f"Number of prompts: {num_prompts}")
        self.info("=" * 80)

    def log_benchmark_end(
        self,
        success: bool,
        duration: float,
        successful_prompts: int,
        total_prompts: int,
    ) -> None:
        """Log benchmark completion.

        Args:
            success: Whether benchmark completed successfully
            duration: Total duration in seconds
            successful_prompts: Number of successful prompts
            total_prompts: Total number of prompts
        """
        self.info("=" * 80)
        if success:
            self.info("Benchmark completed successfully")
        else:
            self.error("Benchmark completed with errors")

        self.info(f"Total duration: {duration:.2f}s")
        self.info(f"Successful prompts: {successful_prompts}/{total_prompts}")
        self.info("=" * 80)

    def log_prompt_start(self, prompt_num: int, total_prompts: int, prompt: str) -> None:
        """Log prompt processing start.

        Args:
            prompt_num: Current prompt number (1-indexed)
            total_prompts: Total number of prompts
            prompt: Prompt text (will be truncated for logging)
        """
        truncated_prompt = prompt[:100] + "..." if len(prompt) > 100 else prompt
        self.info(f"Processing prompt {prompt_num}/{total_prompts}: {truncated_prompt}")

    def log_prompt_end(
        self, prompt_num: int, success: bool, duration: float, tokens: int = 0
    ) -> None:
        """Log prompt processing completion.

        Args:
            prompt_num: Prompt number
            success: Whether prompt was successful
            duration: Processing duration in seconds
            tokens: Number of tokens generated
        """
        if success:
            self.info(
                f"  Prompt {prompt_num} completed in {duration:.2f}s ({tokens} tokens)"
            )
        else:
            self.error(f"  Prompt {prompt_num} failed after {duration:.2f}s")

    def log_error_details(self, error: Exception) -> None:
        """Log detailed error information.

        Args:
            error: Exception object
        """
        self.error(f"Error: {type(error).__name__}: {str(error)}")
        import traceback

        self.debug("Traceback:")
        for line in traceback.format_exc().split("\n"):
            if line.strip():
                self.debug(f"  {line}")

    def log_results(self, results_dict: dict) -> None:
        """Log benchmark results.

        Args:
            results_dict: Results dictionary
        """
        self.info("Benchmark Results:")
        for key, value in results_dict.items():
            if isinstance(value, float):
                self.info(f"  {key}: {value:.4f}")
            else:
                self.info(f"  {key}: {value}")

    def get_log_file_path(self) -> str:
        """Get path to log file.

        Returns:
            Path to log file as string
        """
        return str(self.log_file_path)

    def close(self) -> None:
        """Close all log handlers."""
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)


def main():
    """Test the debug logger."""
    import time

    print("Testing Debug Logger")
    print("=" * 80)

    # Create logger
    logger = DebugLogger(
        log_dir="./test_logs",
        model_name="openai/gpt-oss-20b",
        reasoning_state="On (High)",
    )

    # Test various log levels
    logger.log_benchmark_start(
        model_id="openai/gpt-oss-20b", num_prompts=3, backend="vllm"
    )

    logger.log_config(
        {
            "model_class": "B",
            "task": "text_gen",
            "reasoning_state": "On (High)",
            "use_harmony": True,
            "reasoning_params": {"reasoning_effort": "high"},
        }
    )

    # Simulate prompt processing
    for i in range(1, 4):
        logger.log_prompt_start(i, 3, f"This is test prompt number {i}")
        time.sleep(0.1)
        logger.log_prompt_end(i, True, 0.5, 150)

    # Test error logging
    try:
        raise ValueError("This is a test error")
    except Exception as e:
        logger.log_error_details(e)

    # Log results
    logger.log_results(
        {
            "total_duration": 1.5,
            "avg_latency": 0.5,
            "total_tokens": 450,
            "throughput": 300.0,
        }
    )

    logger.log_benchmark_end(True, 1.5, 3, 3)

    print(f"\nLog file created at: {logger.get_log_file_path()}")

    logger.close()


if __name__ == "__main__":
    main()
