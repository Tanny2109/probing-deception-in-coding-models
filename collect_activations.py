"""
Collect activations from Qwen2.5-Coder model for honest vs deceptive prompts.

This script:
1. Loads the model
2. Runs paired (honest, deceptive) prompts through the model
3. Captures hidden states at the last token position before generation
4. Saves activations with labels for probe training
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
from tqdm import tqdm
import json

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config, ExperimentConfig
from data.prompts import generate_prompts, format_chat_prompt, TaskSample


@dataclass
class ActivationData:
    """Container for collected activations"""
    activations: Dict[int, np.ndarray]  # layer -> (n_samples, hidden_size)
    labels: np.ndarray  # 0 = honest, 1 = deceptive
    categories: List[str]
    task_ids: List[str]


class ActivationCollector:
    """Collects hidden state activations from transformer layers."""

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        layers: Optional[List[int]] = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.layers = layers
        self.activations: Dict[int, List[torch.Tensor]] = {}
        self.hooks = []

    def _get_activation_hook(self, layer_idx: int):
        """Create a hook function for a specific layer."""
        def hook(module, input, output):
            # output is typically (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            # Store the last token's activation
            last_token_activation = hidden_states[:, -1, :].detach().cpu()

            if layer_idx not in self.activations:
                self.activations[layer_idx] = []
            self.activations[layer_idx].append(last_token_activation)

        return hook

    def register_hooks(self):
        """Register forward hooks on model layers."""
        # Get all transformer layers
        if hasattr(self.model, 'model'):
            # For models with model.model.layers structure (Qwen, Llama, etc.)
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            # For GPT-2 style models
            transformer_layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        n_layers = len(transformer_layers)
        layers_to_hook = self.layers if self.layers else list(range(n_layers))

        for layer_idx in layers_to_hook:
            if layer_idx < n_layers:
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._get_activation_hook(layer_idx)
                )
                self.hooks.append(hook)

        print(f"Registered hooks on {len(self.hooks)} layers")

    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}

    def collect_activation(self, prompt: str) -> None:
        """Run a single prompt and collect activations."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        with torch.no_grad():
            # Just do a forward pass, hooks will capture activations
            _ = self.model(**inputs)

    def get_activations(self) -> Dict[int, torch.Tensor]:
        """Get collected activations as stacked tensors."""
        return {
            layer: torch.cat(acts, dim=0)
            for layer, acts in self.activations.items()
        }


def load_model(config: ExperimentConfig) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load the model and tokenizer."""
    print(f"Loading model: {config.model.model_name}")
    print(f"Cache directory: {config.model.cache_dir}")

    # Set dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    dtype = dtype_map.get(config.model.dtype, torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained(
        config.model.model_name,
        cache_dir=str(config.model.cache_dir),
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        config.model.model_name,
        cache_dir=str(config.model.cache_dir),
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True
    )
    model.eval()

    print(f"Model loaded on {model.device}")
    return model, tokenizer


def collect_all_activations(
    config: ExperimentConfig,
    samples: List[TaskSample],
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
) -> ActivationData:
    """
    Collect activations for all samples in both conditions.

    Args:
        config: Experiment configuration
        samples: List of task samples
        model: The loaded model
        tokenizer: The tokenizer

    Returns:
        ActivationData containing all activations and labels
    """
    # Determine which layers to collect
    if config.probe.layers_to_probe:
        layers = config.probe.layers_to_probe
    else:
        layers = list(range(config.model.n_layers))

    collector = ActivationCollector(model, tokenizer, layers)
    collector.register_hooks()

    all_labels = []
    all_categories = []
    all_task_ids = []

    print("Collecting activations...")

    for sample in tqdm(samples, desc="Processing samples"):
        # Collect HONEST activation
        honest_prompt = format_chat_prompt(
            sample.honest_system,
            sample.user_prompt,
            config.model.model_name
        )
        collector.collect_activation(honest_prompt)
        all_labels.append(0)  # 0 = honest
        all_categories.append(sample.category)
        all_task_ids.append(sample.task_id)

        # Collect DECEPTIVE activation
        deceptive_prompt = format_chat_prompt(
            sample.deceptive_system,
            sample.user_prompt,
            config.model.model_name
        )
        collector.collect_activation(deceptive_prompt)
        all_labels.append(1)  # 1 = deceptive
        all_categories.append(sample.category)
        all_task_ids.append(sample.task_id)

    collector.remove_hooks()

    # Get final activations
    activations = collector.get_activations()

    # Convert to numpy (convert to float32 first if bfloat16)
    activations_np = {}
    for layer, acts in activations.items():
        # Convert bfloat16 to float32 before numpy conversion
        if acts.dtype == torch.bfloat16:
            acts = acts.float()
        activations_np[layer] = acts.numpy()

    return ActivationData(
        activations=activations_np,
        labels=np.array(all_labels),
        categories=all_categories,
        task_ids=all_task_ids,
    )


def save_activations(data: ActivationData, output_dir: Path):
    """Save collected activations to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save activations per layer
    for layer, acts in data.activations.items():
        np.save(output_dir / f"activations_layer_{layer:02d}.npy", acts)

    # Save labels and metadata
    np.save(output_dir / "labels.npy", data.labels)

    metadata = {
        "categories": data.categories,
        "task_ids": data.task_ids,
        "n_samples": len(data.labels),
        "layers": list(data.activations.keys()),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"Saved activations to {output_dir}")
    print(f"  - {len(data.labels)} samples")
    print(f"  - {len(data.activations)} layers")
    print(f"  - Shape per layer: {list(data.activations.values())[0].shape}")


def main():
    """Main function to collect activations."""
    config = get_config()

    # Generate prompts
    print("Generating prompts...")
    samples = generate_prompts(
        categories=config.data.categories,
        n_per_category=config.data.n_prompts // len(config.data.categories),
        seed=config.data.seed
    )
    print(f"Generated {len(samples)} task samples")

    # Load model
    model, tokenizer = load_model(config)

    # Collect activations
    data = collect_all_activations(config, samples, model, tokenizer)

    # Save
    save_activations(data, config.activations_dir)

    print("\nDone! Next step: python train_probe.py")


if __name__ == "__main__":
    main()
