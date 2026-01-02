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
    Uses PER-LAYER directions for more accurate steering.
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        tokenizer: AutoTokenizer,
        steering_vectors: Dict[int, np.ndarray],  # layer -> direction
        target_layers: List[int],
        steering_strength: float = 1.0,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.steering_vectors_np = {
            layer: vec for layer, vec in steering_vectors.items()
            if layer in target_layers
        }
        self.target_layers = target_layers
        self.steering_strength = steering_strength
        self.hooks = []
        self.active = False

        # Get transformer layers to determine device for each layer
        if hasattr(model, 'model'):
            self.transformer_layers = model.model.layers
        elif hasattr(model, 'transformer'):
            self.transformer_layers = model.transformer.h
        else:
            raise ValueError("Unknown model architecture")

        # Convert vectors to tensors on the correct device for each layer
        self.steering_vectors = {}
        for layer in target_layers:
            if layer in self.steering_vectors_np and layer < len(self.transformer_layers):
                # Get the device of this specific layer
                layer_device = next(self.transformer_layers[layer].parameters()).device
                self.steering_vectors[layer] = torch.tensor(
                    self.steering_vectors_np[layer],
                    dtype=model.dtype,
                    device=layer_device
                )

    def _get_steering_hook(self, layer_idx: int, add: bool = True):
        """Create a hook that adds/subtracts the steering vector for this layer."""
        multiplier = self.steering_strength if add else -self.steering_strength
        layer_vector = self.steering_vectors[layer_idx]

        def hook(module, input, output):
            if not self.active:
                return output

            if isinstance(output, tuple):
                hidden_states = output[0]
                # Add steering vector to all token positions
                modified = hidden_states + multiplier * layer_vector
                return (modified,) + output[1:]
            else:
                return output + multiplier * layer_vector

        return hook

    def register_hooks(self, add: bool = True):
        """Register forward hooks for steering."""
        self.remove_hooks()

        for layer_idx in self.target_layers:
            if layer_idx in self.steering_vectors:
                hook = self.transformer_layers[layer_idx].register_forward_hook(
                    self._get_steering_hook(layer_idx, add=add)
                )
                self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        do_sample: bool = False,  # Greedy for speed + determinism
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


def load_steering_resources(config: ExperimentConfig) -> Tuple[Dict[int, np.ndarray], int]:
    """Load deception vectors for all layers and determine best layer."""
    directions_dir = config.probes_dir / "directions"
    results_path = config.probes_dir / "paired_analysis_results.json"

    if not directions_dir.exists():
        raise FileNotFoundError(
            "No deception directions found. Run analyze_paired_differences.py first."
        )

    # Load results to find best layer
    with open(results_path) as f:
        results = json.load(f)

    best_layer = results.get("best_layer_separability", 18)
    print(f"Best layer by separability: {best_layer}")

    # Load direction for each layer
    deception_vectors = {}
    for layer_file in directions_dir.glob("deception_direction_layer_*.npy"):
        layer_num = int(layer_file.stem.split("_")[-1])
        deception_vectors[layer_num] = np.load(layer_file)
        print(f"  Loaded direction for layer {layer_num}")

    return deception_vectors, best_layer


def run_steering_experiment(
    config: ExperimentConfig,
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    samples: List[TaskSample],
    deception_vectors: Dict[int, np.ndarray],  # layer -> direction
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
            model, tokenizer, deception_vectors, target_layers,
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
        for strength in config.steering.steering_strengths:
            steerer = ActivationSteerer(
                model, tokenizer, deception_vectors, target_layers,
                steering_strength=strength
            )
            steerer.register_hooks(add=True)
            gen_steered = steerer.generate(honest_prompt)
            steerer.remove_hooks()

            results.append(SteeringResult(
                task_id=sample.task_id,
                category=sample.category,
                condition=f"honest+deception_s{strength}",
                prompt=sample.user_prompt,
                generation=gen_steered,
                steering_strength=strength
            ))

        # 4. Deceptive - deception vector (should prevent deceptive behavior)
        for strength in config.steering.steering_strengths:
            steerer = ActivationSteerer(
                model, tokenizer, deception_vectors, target_layers,
                steering_strength=strength
            )
            steerer.register_hooks(add=False)  # Subtract
            gen_steered = steerer.generate(deceptive_prompt)
            steerer.remove_hooks()

            results.append(SteeringResult(
                task_id=sample.task_id,
                category=sample.category,
                condition=f"deceptive-deception_s{strength}",
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

        # Get any result to show the prompt
        any_result = list(task_results.values())[0]
        print(f"Prompt: {any_result.prompt[:100]}...")

        # Show baselines first, then steered results by strength
        for condition in sorted(task_results.keys()):
            r = task_results[condition]
            print(f"\n[{condition.upper()}] (strength={r.steering_strength}):")
            # Show first 300 chars of generation
            gen_preview = r.generation[:300].replace('\n', ' ')
            print(f"  {gen_preview}...")

    print("\n" + "="*80)


def print_summary_stats(results: List[SteeringResult]):
    """Print summary statistics about the steering results."""
    from collections import defaultdict

    # Group by condition
    by_condition = defaultdict(list)
    for r in results:
        by_condition[r.condition].append(r)

    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)

    for condition in sorted(by_condition.keys()):
        samples = by_condition[condition]
        avg_len = sum(len(s.generation) for s in samples) / len(samples)
        print(f"{condition}: {len(samples)} samples, avg generation length: {avg_len:.0f} chars")


def main():
    """Main steering experiment function."""
    config = get_config()

    # Check if directions exist
    directions_dir = config.probes_dir / "directions"
    if not directions_dir.exists():
        print("Error: No deception directions found.")
        print("Please run analyze_paired_differences.py first.")
        return

    # Load resources
    print("Loading deception vectors and model...")
    deception_vectors, best_layer = load_steering_resources(config)

    # Determine target layers (use best layer and neighbors)
    target_layers = [best_layer - 1, best_layer, best_layer + 1]
    target_layers = [l for l in target_layers if l in deception_vectors]

    print(f"Best layer: {best_layer}")
    print(f"Steering target layers: {target_layers}")

    # Load model
    from collect_activations import load_model
    model, tokenizer = load_model(config)

    # Generate test samples - fewer samples but full code generation
    print("\nGenerating test samples...")
    samples = generate_prompts(
        categories=config.data.categories,
        n_per_category=5,  # 30 total samples (5 * 6 categories)
        seed=config.data.seed + 100  # Different seed for test
    )

    n_strengths = len(config.steering.steering_strengths)
    n_conditions = 2 + n_strengths * 2  # 2 baselines + strengths * 2 directions

    # Run experiments
    print(f"\nRunning steering experiments on {len(samples)} samples...")
    print(f"Steering strengths: {config.steering.steering_strengths}")
    print(f"Target layers: {target_layers}")
    print(f"Conditions per sample: {n_conditions} (2 baselines + {n_strengths} strengths × 2 directions)")
    print(f"Total generations: {len(samples) * n_conditions}")

    results = run_steering_experiment(
        config, model, tokenizer, samples,
        deception_vectors, target_layers,
        n_samples=30  # Use all 30 samples
    )

    # Save results
    save_steering_results(
        results,
        config.results_dir / "steering_results.json"
    )

    # Print comparisons and stats
    print_sample_comparisons(results, n=3)
    print_summary_stats(results)

    print("\nSteering experiment complete!")
    print(f"Results saved to: {config.results_dir / 'steering_results.json'}")
    print("\nKey questions to analyze:")
    print("  1. Does adding deception vector to honest prompts produce vulnerable code?")
    print("  2. Does subtracting deception vector from deceptive prompts fix the code?")
    print("  3. Is the effect consistent across vulnerability categories?")


if __name__ == "__main__":
    main()
