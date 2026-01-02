"""
Activation steering experiments.

This script tests whether we can:
1. ADD deception vector to honest prompts -> induce deceptive behavior
2. SUBTRACT deception vector from deceptive prompts -> prevent deceptive behavior

This is the "causal intervention" that proves the vector actually
represents deceptive intent, not just correlation.
"""

import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
import json
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer

from config import get_config, ExperimentConfig
from data.prompts import generate_prompts, format_chat_prompt, TaskSample


@dataclass
class SteeringResult:
    """Result of a single steering experiment"""
    task_id: str
    category: str
    condition: str  # "honest", "deceptive", "honest+deception", "deceptive-deception"
    prompt: str
    generation: str
    steering_strength: float


class ActivationSteerer:
    """
    Steers model activations during generation.

    Adds or subtracts a direction vector from hidden states
    at specified layers during the forward pass.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        steering_vector: np.ndarray,
        target_layers: List[int],
        steering_strength: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vector = torch.tensor(
            steering_vector, dtype=model.dtype, device=model.device
        )
        self.target_layers = target_layers
        self.steering_strength = steering_strength
        self.hooks = []
        self.active = False

    def _get_steering_hook(self, layer_idx: int, add: bool = True):
        """Create a hook that adds/subtracts the steering vector."""
        multiplier = self.steering_strength if add else -self.steering_strength

        def hook(module, input, output):
            if not self.active:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add steering vector to all token positions
                modified = hidden_states + multiplier * self.steering_vector
                return (modified,) + output[1:]
            else:
                return output + multiplier * self.steering_vector

        return hook

    def register_hooks(self, add: bool = True):
        """Register forward hooks for steering."""
        self.remove_hooks()

        # Get transformer layers
        if hasattr(self.model, 'model'):
            transformer_layers = self.model.model.layers
        elif hasattr(self.model, 'transformer'):
            transformer_layers = self.model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        for layer_idx in self.target_layers:
            if layer_idx < len(transformer_layers):
                hook = transformer_layers[layer_idx].register_forward_hook(
                    self._get_steering_hook(layer_idx, add=add)
                )
                self.hooks.append(hook)

        print(f"Registered steering hooks on layers {self.target_layers}")

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        do_sample: bool = True,
    ) -> str:
        """Generate text with steering active."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)

        self.active = True
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        self.active = False

        # Decode only the new tokens
        generated_ids = outputs[0][inputs['input_ids'].shape[1]:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        return generated_text


def load_steering_resources(config: ExperimentConfig) -> Tuple[np.ndarray, int]:
    """Load deception vector and determine best layer."""
    # Load deception direction
    deception_vector = np.load(config.probes_dir / "deception_direction.npy")

    # Load probe results to find best layer
    with open(config.probes_dir / "probe_results.json") as f:
        probe_results = json.load(f)

    best_layer = int(max(probe_results.keys(),
                        key=lambda l: probe_results[l]['cv_mean']))

    return deception_vector, best_layer


def run_steering_experiment(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[TaskSample],
    deception_vector: np.ndarray,
    target_layers: List[int],
    n_samples: int = 10,
) -> List[SteeringResult]:
    """
    Run the full steering experiment.

    For each sample:
    1. Generate with honest prompt (baseline)
    2. Generate with deceptive prompt (baseline)
    3. Generate with honest prompt + deception vector (steering test)
    4. Generate with deceptive prompt - deception vector (steering test)
    """
    results = []
    samples_to_test = samples[:n_samples]

    for sample in tqdm(samples_to_test, desc="Running steering experiments"):
        # Format prompts
        honest_prompt = format_chat_prompt(
            sample.honest_system,
            sample.user_prompt,
            config.model.model_name
        )
        deceptive_prompt = format_chat_prompt(
            sample.deceptive_system,
            sample.user_prompt,
            config.model.model_name
        )

        # 1. Baseline honest generation
        steerer = ActivationSteerer(
            model, tokenizer, deception_vector, target_layers,
            steering_strength=0.0  # No steering
        )
        steerer.register_hooks(add=True)
        gen_honest = steerer.generate(honest_prompt)
        steerer.remove_hooks()

        results.append(SteeringResult(
            task_id=sample.task_id,
            category=sample.category,
            condition="honest_baseline",
            prompt=sample.user_prompt,
            generation=gen_honest,
            steering_strength=0.0
        ))

        # 2. Baseline deceptive generation
        gen_deceptive = steerer.generate(deceptive_prompt)

        results.append(SteeringResult(
            task_id=sample.task_id,
            category=sample.category,
            condition="deceptive_baseline",
            prompt=sample.user_prompt,
            generation=gen_deceptive,
            steering_strength=0.0
        ))

        # 3. Honest + deception vector (should induce deceptive behavior)
        for strength in config.steering.steering_strength if isinstance(
            config.steering.steering_strength, list
        ) else [config.steering.steering_strength]:

            steerer = ActivationSteerer(
                model, tokenizer, deception_vector, target_layers,
                steering_strength=strength
            )
            steerer.register_hooks(add=True)
            gen_steered = steerer.generate(honest_prompt)
            steerer.remove_hooks()

            results.append(SteeringResult(
                task_id=sample.task_id,
                category=sample.category,
                condition="honest+deception",
                prompt=sample.user_prompt,
                generation=gen_steered,
                steering_strength=strength
            ))

        # 4. Deceptive - deception vector (should prevent deceptive behavior)
        for strength in config.steering.steering_strength if isinstance(
            config.steering.steering_strength, list
        ) else [config.steering.steering_strength]:

            steerer = ActivationSteerer(
                model, tokenizer, deception_vector, target_layers,
                steering_strength=strength
            )
            steerer.register_hooks(add=False)  # Subtract
            gen_steered = steerer.generate(deceptive_prompt)
            steerer.remove_hooks()

            results.append(SteeringResult(
                task_id=sample.task_id,
                category=sample.category,
                condition="deceptive-deception",
                prompt=sample.user_prompt,
                generation=gen_steered,
                steering_strength=strength
            ))

    return results


def save_steering_results(results: List[SteeringResult], output_path: Path):
    """Save steering experiment results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = [
        {
            'task_id': r.task_id,
            'category': r.category,
            'condition': r.condition,
            'prompt': r.prompt,
            'generation': r.generation,
            'steering_strength': r.steering_strength
        }
        for r in results
    ]

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Saved {len(results)} steering results to {output_path}")


def print_sample_comparisons(results: List[SteeringResult], n: int = 3):
    """Print side-by-side comparisons of steering effects."""
    # Group by task_id
    by_task = {}
    for r in results:
        if r.task_id not in by_task:
            by_task[r.task_id] = {}
        by_task[r.task_id][r.condition] = r

    task_ids = list(by_task.keys())[:n]

    print("\n" + "="*80)
    print("STEERING EXPERIMENT RESULTS")
    print("="*80)

    for task_id in task_ids:
        task_results = by_task[task_id]
        print(f"\n{'─'*80}")
        print(f"Task: {task_id}")
        print(f"Prompt: {task_results.get('honest_baseline', task_results.get('deceptive_baseline')).prompt[:100]}...")

        for condition in ['honest_baseline', 'deceptive_baseline', 'honest+deception', 'deceptive-deception']:
            if condition in task_results:
                r = task_results[condition]
                print(f"\n[{condition.upper()}] (strength={r.steering_strength}):")
                # Show first 300 chars of generation
                gen_preview = r.generation[:300].replace('\n', ' ')
                print(f"  {gen_preview}...")

    print("\n" + "="*80)


def main():
    """Main steering experiment function."""
    config = get_config()

    # Check if probes exist
    if not (config.probes_dir / "deception_direction.npy").exists():
        print("Error: No deception direction found.")
        print("Please run train_probe.py first.")
        return

    # Load resources
    print("Loading deception vector and model...")
    deception_vector, best_layer = load_steering_resources(config)

    # Determine target layers (use best layer and neighbors)
    target_layers = [best_layer - 1, best_layer, best_layer + 1]
    target_layers = [l for l in target_layers if 0 <= l < config.model.n_layers]

    print(f"Using deception vector from layer {best_layer}")
    print(f"Steering target layers: {target_layers}")

    # Load model
    from collect_activations import load_model
    model, tokenizer = load_model(config)

    # Generate test samples
    print("\nGenerating test samples...")
    samples = generate_prompts(
        categories=config.data.categories,
        n_per_category=3,  # Fewer for steering experiments (they're slow)
        seed=config.data.seed + 100  # Different seed for test
    )

    # Run experiments
    print("\nRunning steering experiments...")
    results = run_steering_experiment(
        config, model, tokenizer, samples,
        deception_vector, target_layers,
        n_samples=10
    )

    # Save results
    save_steering_results(
        results,
        config.results_dir / "steering_results.json"
    )

    # Print comparisons
    print_sample_comparisons(results, n=5)

    print("\nSteering experiment complete!")
    print(f"Results saved to: {config.results_dir / 'steering_results.json'}")
    print("\nKey questions to analyze:")
    print("  1. Does adding deception vector to honest prompts produce vulnerable code?")
    print("  2. Does subtracting deception vector from deceptive prompts fix the code?")
    print("  3. Is the effect consistent across vulnerability categories?")


if __name__ == "__main__":
    main()
