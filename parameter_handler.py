#!/usr/bin/env python3
"""
Parameter Handler for AI Energy Score Batch Runner.

Handles model-specific parameter processing, prompt formatting,
and reasoning configuration for different model types.
"""

from typing import Any, Dict, Optional

from model_config_parser import ModelConfig


class ParameterHandler:
    """Handles model-specific parameters and prompt formatting."""

    @staticmethod
    def prepare_prompt(config: ModelConfig, original_prompt: str) -> str:
        """Prepare prompt with model-specific prefixes/suffixes.

        Args:
            config: ModelConfig with prompt modifications
            original_prompt: Original prompt text

        Returns:
            Modified prompt with prefixes/suffixes applied
        """
        prompt = original_prompt

        # Apply prefix
        if config.prompt_prefix:
            prompt = config.prompt_prefix + prompt

        # Apply suffix
        if config.prompt_suffix:
            prompt = prompt + config.prompt_suffix

        return prompt

    @staticmethod
    def get_backend_config(config: ModelConfig) -> Dict[str, Any]:
        """Get backend configuration for the model.

        Args:
            config: ModelConfig with backend settings

        Returns:
            Dict with backend configuration
        """
        backend_config = {
            "model": config.model_id,
            "use_harmony": config.use_harmony,
        }

        return backend_config

    @staticmethod
    def get_generation_kwargs(
        config: ModelConfig, base_kwargs: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Get generation kwargs including reasoning parameters.

        Args:
            config: ModelConfig with reasoning parameters
            base_kwargs: Base generation kwargs (max_tokens, temperature, etc.)

        Returns:
            Dict with generation kwargs including reasoning params
        """
        kwargs = base_kwargs.copy() if base_kwargs else {}

        # Add reasoning parameters if present
        if config.reasoning_params:
            kwargs["reasoning_params"] = config.reasoning_params

        return kwargs

    @staticmethod
    def get_model_info(config: ModelConfig) -> Dict[str, Any]:
        """Get comprehensive model information for logging/reporting.

        Args:
            config: ModelConfig to extract info from

        Returns:
            Dict with model information
        """
        return {
            "model_id": config.model_id,
            "model_class": config.model_class,
            "task": config.task,
            "reasoning_state": config.reasoning_state,
            "use_harmony": config.use_harmony,
            "reasoning_params": config.reasoning_params,
            "prompt_prefix": config.prompt_prefix,
            "prompt_suffix": config.prompt_suffix,
            "priority": config.priority,
        }

    @staticmethod
    def format_model_name_for_filename(model_id: str, reasoning_state: str) -> str:
        """Format model name for use in filenames.

        Args:
            model_id: HuggingFace model ID
            reasoning_state: Reasoning state string

        Returns:
            Sanitized filename-safe string
        """
        # Replace slashes and spaces with underscores
        safe_model = model_id.replace("/", "_").replace(" ", "_")

        # Sanitize reasoning state
        safe_reasoning = (
            reasoning_state.replace(" ", "_")
            .replace("(", "")
            .replace(")", "")
            .replace("/", "_")
        )

        return f"{safe_model}_{safe_reasoning}"

    @staticmethod
    def validate_config(config: ModelConfig) -> tuple[bool, Optional[str]]:
        """Validate model configuration.

        Args:
            config: ModelConfig to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if not config.model_id:
            return False, "Missing model_id"

        if not config.task:
            return False, "Missing task"

        # Validate class if present
        if config.model_class and config.model_class.upper() not in ["A", "B", "C", ""]:
            return False, f"Invalid model class: {config.model_class}"

        # Validate priority
        if config.priority < 1 or config.priority > 3:
            return False, f"Invalid priority: {config.priority}"

        # Check for conflicting settings
        if config.use_harmony and config.prompt_prefix:
            # This is OK - some models might have both
            pass

        return True, None


def main():
    """Test the parameter handler."""
    from model_config_parser import ModelConfig

    # Test gpt-oss model
    print("=" * 80)
    print("Testing Parameter Handler")
    print("=" * 80)

    # Test 1: gpt-oss model with Harmony
    print("\n1. gpt-oss model with Harmony:")
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

    prompt = ParameterHandler.prepare_prompt(config1, "What is quantum computing?")
    backend_config = ParameterHandler.get_backend_config(config1)
    gen_kwargs = ParameterHandler.get_generation_kwargs(
        config1, {"max_tokens": 100, "temperature": 0.7}
    )

    print(f"  Prompt: {prompt}")
    print(f"  Backend Config: {backend_config}")
    print(f"  Generation Kwargs: {gen_kwargs}")

    # Test 2: DeepSeek with <think> prefix
    print("\n2. DeepSeek model with <think> prefix:")
    config2 = ModelConfig(
        model_id="deepseek-ai/DeepSeek-R1",
        priority=1,
        model_class="C",
        task="text_gen",
        reasoning_state="On",
        chat_template="Prepend input with <think>",
        prompt_prefix="<think>",
    )

    prompt2 = ParameterHandler.prepare_prompt(config2, "Explain AI")
    filename2 = ParameterHandler.format_model_name_for_filename(
        config2.model_id, config2.reasoning_state
    )

    print(f"  Prompt: {prompt2}")
    print(f"  Filename: {filename2}")

    # Test 3: Validation
    print("\n3. Validation tests:")
    is_valid, error = ParameterHandler.validate_config(config1)
    print(f"  Config1 valid: {is_valid}, Error: {error}")

    invalid_config = ModelConfig(
        model_id="",
        priority=1,
        model_class="X",
        task="",
        reasoning_state="",
        chat_template="",
    )
    is_valid, error = ParameterHandler.validate_config(invalid_config)
    print(f"  Invalid config valid: {is_valid}, Error: {error}")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
