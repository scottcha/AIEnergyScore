#!/usr/bin/env python3
"""
Model Configuration Parser for AI Energy Score Batch Runner.

Parses the "AI Energy Score (Oct 2025) - Models.csv" file and extracts
model configurations including reasoning parameters and special formatting requirements.
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


@dataclass
class ModelConfig:
    """Configuration for a single model benchmark run."""

    model_id: str  # HuggingFace model ID
    priority: int  # Priority level (1=high, 2=medium, 3=low)
    model_class: str  # Class A/B/C
    task: str  # Task type (text_gen, image_gen, etc.)
    reasoning_state: str  # Reasoning state (On, Off, On (High), etc.)
    chat_template: str  # Raw chat template/parameters from CSV
    reasoning_params: Optional[Dict[str, Any]] = None  # Parsed reasoning parameters
    use_harmony: bool = False  # Enable Harmony formatting (gpt-oss models)
    prompt_prefix: str = ""  # Prefix to add to prompts
    prompt_suffix: str = ""  # Suffix to add to prompts


class ModelConfigParser:
    """Parser for AI Energy Score model configuration CSV."""

    def __init__(self, csv_path: str):
        """Initialize parser with CSV file path.

        Args:
            csv_path: Path to the CSV file
        """
        self.csv_path = Path(csv_path)
        if not self.csv_path.exists():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

    def parse(self) -> List[ModelConfig]:
        """Parse CSV file and return list of model configurations.

        Returns:
            List of ModelConfig objects
        """
        # Read CSV file
        df = pd.read_csv(self.csv_path)

        # Clean column names (remove leading/trailing spaces)
        df.columns = df.columns.str.strip()

        # Validate required columns
        required_columns = [
            "Models to Add",
            "Priority (1=hi)",
            "Class",
            "Task",
            "Reasoning State",
            "Chat Template",
        ]
        missing_columns = set(required_columns) - set(df.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Parse each row
        configs = []
        for _, row in df.iterrows():
            # Skip rows with empty model ID
            if pd.isna(row["Models to Add"]) or not str(row["Models to Add"]).strip():
                continue

            # Extract basic fields
            model_id = str(row["Models to Add"]).strip()
            # Remove https://huggingface.co/ prefix if present
            model_id = model_id.replace("https://huggingface.co/", "")

            # Parse priority (default to 1 if missing)
            try:
                priority = int(row["Priority (1=hi)"])
            except (ValueError, TypeError):
                priority = 1

            model_class = str(row["Class"]).strip() if not pd.isna(row["Class"]) else ""
            task = str(row["Task"]).strip() if not pd.isna(row["Task"]) else ""
            reasoning_state = (
                str(row["Reasoning State"]).strip()
                if not pd.isna(row["Reasoning State"])
                else ""
            )
            chat_template = (
                str(row["Chat Template"]).strip()
                if not pd.isna(row["Chat Template"])
                else ""
            )

            # Parse model-specific parameters
            config = ModelConfig(
                model_id=model_id,
                priority=priority,
                model_class=model_class,
                task=task,
                reasoning_state=reasoning_state,
                chat_template=chat_template,
            )

            # Parse chat template to extract reasoning params and flags
            self._parse_chat_template(config)

            configs.append(config)

        return configs

    def _parse_chat_template(self, config: ModelConfig) -> None:
        """Parse chat template and populate reasoning params and flags.

        Args:
            config: ModelConfig to populate (modified in-place)
        """
        template = config.chat_template.lower()
        model_id_lower = config.model_id.lower()

        # Skip if N/A or empty
        if not template or "n/a" in template:
            return

        # gpt-oss models: Enable Harmony formatting
        if "gpt-oss" in model_id_lower:
            config.use_harmony = True

            # Extract reasoning effort
            if "reasoning_effort: high" in template or "reasoning: high" in template:
                config.reasoning_params = {"reasoning_effort": "high"}
            elif "reasoning_effort: low" in template or "reasoning: low" in template:
                config.reasoning_params = {"reasoning_effort": "low"}
            elif "reasoning_effort: medium" in template or "reasoning: medium" in template:
                config.reasoning_params = {"reasoning_effort": "medium"}
            elif "reasoning: true" in template:
                # Default to high if just "reasoning: true"
                config.reasoning_params = {"reasoning_effort": "high"}

        # DeepSeek models: Prepend <think> for thinking mode
        elif "deepseek" in model_id_lower:
            if "<think>" in template or "prepend input with <think>" in template:
                config.prompt_prefix = "<think>"
            elif "enable_thinking=true" in template:
                config.reasoning_params = {"enable_thinking": True}
            elif "enable_thinking=false" in template:
                config.reasoning_params = {"enable_thinking": False}

        # Qwen models: enable_thinking parameter
        elif "qwen" in model_id_lower:
            if "enable_thinking=true" in template:
                config.reasoning_params = {"enable_thinking": True}
            elif "enable_thinking=false" in template:
                config.reasoning_params = {"enable_thinking": False}
            # Some Qwen models have thinking mode by default
            elif "thinking mode" in template or "supports only thinking mode" in template:
                config.prompt_prefix = "<think>"

        # Hunyuan models: /think prefix
        elif "hunyuan" in model_id_lower:
            if "/think" in template:
                config.prompt_prefix = "/think "
            elif "enable_thinking=false" in template:
                config.reasoning_params = {"enable_thinking": False}

        # EXAONE models: Inverted logic (Off = thinking on)
        elif "exaone" in model_id_lower:
            if "enable_thinking=true" in template:
                config.reasoning_params = {"enable_thinking": True}
            elif "enable_thinking=false" in template:
                config.reasoning_params = {"enable_thinking": False}

        # Nemotron models: /no_think to disable
        elif "nemotron" in model_id_lower:
            if "/no_think" in template:
                config.prompt_prefix = "/no_think "
            # Default is thinking ON, so no action needed for "On"

        # Phi models: Generic reasoning parameter
        elif "phi" in model_id_lower:
            if "reasoning: true" in template:
                config.reasoning_params = {"reasoning": True}
            elif "reasoning: false" in template:
                config.reasoning_params = {"reasoning": False}

        # SmolLM models: Generic reasoning parameter
        elif "smollm" in model_id_lower:
            if "reasoning: true" in template:
                config.reasoning_params = {"reasoning": True}
            elif "reasoning: false" in template:
                config.reasoning_params = {"reasoning": False}

        # Gemma models: Generic reasoning parameter
        elif "gemma" in model_id_lower:
            if "reasoning: true" in template:
                config.reasoning_params = {"reasoning": True}

        # Generic fallback: Look for common patterns
        else:
            # Check for generic reasoning parameter
            if "reasoning: true" in template:
                config.reasoning_params = {"reasoning": True}
            elif "reasoning: false" in template:
                config.reasoning_params = {"reasoning": False}

    def filter_configs(
        self,
        configs: List[ModelConfig],
        model_name: Optional[str] = None,
        model_class: Optional[str] = None,
        task: Optional[str] = None,
        reasoning_state: Optional[str] = None,
    ) -> List[ModelConfig]:
        """Filter model configurations based on criteria.

        Args:
            configs: List of ModelConfig objects to filter
            model_name: Filter by model name (substring match)
            model_class: Filter by model class (A, B, or C)
            task: Filter by task type
            reasoning_state: Filter by reasoning state

        Returns:
            Filtered list of ModelConfig objects
        """
        filtered = configs

        if model_name:
            filtered = [c for c in filtered if model_name.lower() in c.model_id.lower()]

        if model_class:
            filtered = [
                c for c in filtered if c.model_class.upper() == model_class.upper()
            ]

        if task:
            filtered = [c for c in filtered if task.lower() in c.task.lower()]

        if reasoning_state:
            filtered = [
                c
                for c in filtered
                if reasoning_state.lower() in c.reasoning_state.lower()
            ]

        return filtered


def main():
    """Test the parser with sample CSV."""
    import sys

    csv_path = sys.argv[1] if len(sys.argv) > 1 else "AI Energy Score (Oct 2025) - Models.csv"

    parser = ModelConfigParser(csv_path)
    configs = parser.parse()

    print(f"Parsed {len(configs)} model configurations\n")
    print("=" * 80)

    for i, config in enumerate(configs[:5], 1):  # Show first 5
        print(f"\nModel {i}: {config.model_id}")
        print(f"  Class: {config.model_class}, Task: {config.task}")
        print(f"  Reasoning State: {config.reasoning_state}")
        print(f"  Use Harmony: {config.use_harmony}")
        print(f"  Reasoning Params: {config.reasoning_params}")
        print(f"  Prompt Prefix: '{config.prompt_prefix}'")

    print("\n" + "=" * 80)
    print(f"\n... and {len(configs) - 5} more models")


if __name__ == "__main__":
    main()
