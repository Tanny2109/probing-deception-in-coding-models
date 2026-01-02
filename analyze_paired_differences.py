"""
Analyze deception using PAIRED DIFFERENCES approach.

Instead of classifying raw activations (which picks up prompt text),
we compute: diff = activation_deceptive - activation_honest

For each task, this gives us a "deception direction" - what changes
when the model is told to be deceptive vs honest for the SAME task.

Key questions:
1. Is this direction consistent across tasks?
2. Does it generalize across vulnerability categories?
3. Can we use it for steering?
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from scipy import stats

from config import get_config, ExperimentConfig


@dataclass
class PairedAnalysisResult:
    """Results from paired difference analysis"""
    layer: int
    mean_deception_direction: np.ndarray
    direction_consistency: float  # How similar are individual diffs to mean?
    cross_category_consistency: float  # Does direction generalize?
    projection_separability: float  # Can we separate honest/deceptive by projecting?


def load_activations_paired(
    activations_dir: Path,
) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], List[str], List[str]]:
    """
    Load activations and split into honest/deceptive pairs.

    The data was collected as: [honest_0, deceptive_0, honest_1, deceptive_1, ...]
    So even indices = honest, odd indices = deceptive
    """
    # Load metadata
    with open(activations_dir / "metadata.json") as f:
        metadata = json.load(f)

    labels = np.load(activations_dir / "labels.npy")

    # Load all layers
    activations = {}
    for layer in metadata["layers"]:
        filepath = activations_dir / f"activations_layer_{layer:02d}.npy"
        if filepath.exists():
            activations[layer] = np.load(filepath)

    # Split into honest (even indices) and deceptive (odd indices)
    honest_acts = {layer: acts[::2] for layer, acts in activations.items()}
    deceptive_acts = {layer: acts[1::2] for layer, acts in activations.items()}

    # Get categories for each pair (every other one since we have pairs)
    categories = metadata["categories"][::2]
    task_ids = metadata["task_ids"][::2]

    n_pairs = len(categories)
    print(f"Loaded {n_pairs} paired samples")
    print(f"Honest shape: {list(honest_acts.values())[0].shape}")
    print(f"Deceptive shape: {list(deceptive_acts.values())[0].shape}")

    return honest_acts, deceptive_acts, categories, task_ids


def compute_deception_directions(
    honest_acts: Dict[int, np.ndarray],
    deceptive_acts: Dict[int, np.ndarray],
) -> Dict[int, np.ndarray]:
    """
    Compute deception direction for each sample: deceptive - honest

    Returns dict of layer -> (n_samples, hidden_size) difference vectors
    """
    diffs = {}
    for layer in honest_acts.keys():
        diffs[layer] = deceptive_acts[layer] - honest_acts[layer]
    return diffs


def analyze_direction_consistency(
    diffs: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """
    Analyze how consistent the deception direction is across samples.

    Returns:
        mean_direction: The average deception direction (normalized)
        consistency: Average cosine similarity of individual diffs to mean
    """
    # Compute mean direction
    mean_direction = diffs.mean(axis=0)
    mean_direction = mean_direction / np.linalg.norm(mean_direction)

    # Normalize individual diffs
    norms = np.linalg.norm(diffs, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    diffs_normalized = diffs / norms

    # Compute cosine similarity to mean
    similarities = diffs_normalized @ mean_direction
    consistency = np.mean(similarities)

    return mean_direction, consistency


def analyze_cross_category_generalization(
    diffs: np.ndarray,
    categories: List[str],
    train_categories: List[str],
    test_categories: List[str],
) -> float:
    """
    Test if deception direction learned from some categories
    generalizes to held-out categories.
    """
    categories = np.array(categories)

    # Split by category
    train_mask = np.isin(categories, train_categories)
    test_mask = np.isin(categories, test_categories)

    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return 0.0

    train_diffs = diffs[train_mask]
    test_diffs = diffs[test_mask]

    # Compute mean direction from train categories
    train_mean = train_diffs.mean(axis=0)
    train_mean = train_mean / np.linalg.norm(train_mean)

    # Test how well it aligns with test category diffs
    test_norms = np.linalg.norm(test_diffs, axis=1, keepdims=True)
    test_norms[test_norms == 0] = 1
    test_normalized = test_diffs / test_norms

    similarities = test_normalized @ train_mean
    return np.mean(similarities)


def analyze_projection_separability(
    honest_acts: np.ndarray,
    deceptive_acts: np.ndarray,
    direction: np.ndarray,
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Project honest and deceptive activations onto the deception direction.
    See if they're separable.

    Returns:
        separability: t-statistic of the difference (higher = more separable)
        honest_projections: scalar projections for honest samples
        deceptive_projections: scalar projections for deceptive samples
    """
    honest_proj = honest_acts @ direction
    deceptive_proj = deceptive_acts @ direction

    # t-test for difference
    t_stat, p_value = stats.ttest_ind(deceptive_proj, honest_proj)

    return t_stat, honest_proj, deceptive_proj


def run_paired_analysis(config: ExperimentConfig) -> Dict[int, PairedAnalysisResult]:
    """Run full paired difference analysis."""

    # Load paired data
    honest_acts, deceptive_acts, categories, task_ids = load_activations_paired(
        config.activations_dir
    )

    # Compute differences
    diffs = compute_deception_directions(honest_acts, deceptive_acts)

    results = {}

    print("\n" + "="*60)
    print("PAIRED DIFFERENCE ANALYSIS")
    print("="*60)

    for layer in sorted(diffs.keys()):
        layer_diffs = diffs[layer]

        # Direction consistency
        mean_dir, consistency = analyze_direction_consistency(layer_diffs)

        # Cross-category generalization
        cross_cat = analyze_cross_category_generalization(
            layer_diffs,
            categories,
            config.data.train_categories,
            config.data.test_categories,
        )

        # Projection separability
        t_stat, _, _ = analyze_projection_separability(
            honest_acts[layer],
            deceptive_acts[layer],
            mean_dir,
        )

        results[layer] = PairedAnalysisResult(
            layer=layer,
            mean_deception_direction=mean_dir,
            direction_consistency=consistency,
            cross_category_consistency=cross_cat,
            projection_separability=t_stat,
        )

        print(f"Layer {layer:2d}: consistency={consistency:.3f}, "
              f"cross-cat={cross_cat:.3f}, separability={t_stat:.1f}")

    return results, honest_acts, deceptive_acts, diffs, categories


def plot_results(
    results: Dict[int, PairedAnalysisResult],
    output_dir: Path,
):
    """Generate visualization plots."""
    output_dir.mkdir(parents=True, exist_ok=True)

    layers = sorted(results.keys())
    consistency = [results[l].direction_consistency for l in layers]
    cross_cat = [results[l].cross_category_consistency for l in layers]
    separability = [results[l].projection_separability for l in layers]

    fig, axes = plt.subplots(3, 1, figsize=(12, 10))

    # Plot 1: Direction consistency
    axes[0].plot(layers, consistency, 'o-', linewidth=2, markersize=8)
    axes[0].axhline(y=0, color='gray', linestyle='--')
    axes[0].set_xlabel('Layer')
    axes[0].set_ylabel('Cosine Similarity')
    axes[0].set_title('Direction Consistency (individual diffs vs mean)')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Cross-category generalization
    axes[1].plot(layers, cross_cat, 's-', color='orange', linewidth=2, markersize=8)
    axes[1].axhline(y=0, color='gray', linestyle='--')
    axes[1].set_xlabel('Layer')
    axes[1].set_ylabel('Cosine Similarity')
    axes[1].set_title('Cross-Category Generalization (train→test categories)')
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Projection separability
    axes[2].bar(layers, separability, color='green', alpha=0.7)
    axes[2].axhline(y=0, color='gray', linestyle='--')
    axes[2].set_xlabel('Layer')
    axes[2].set_ylabel('t-statistic')
    axes[2].set_title('Projection Separability (t-test: deceptive vs honest)')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "paired_analysis.png", dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved plot to {output_dir / 'paired_analysis.png'}")


def plot_projection_distribution(
    honest_acts: Dict[int, np.ndarray],
    deceptive_acts: Dict[int, np.ndarray],
    results: Dict[int, PairedAnalysisResult],
    best_layer: int,
    output_dir: Path,
):
    """Plot distribution of projections onto deception direction."""
    direction = results[best_layer].mean_deception_direction

    honest_proj = honest_acts[best_layer] @ direction
    deceptive_proj = deceptive_acts[best_layer] @ direction

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(honest_proj, bins=20, alpha=0.6, label='Honest', color='green')
    ax.hist(deceptive_proj, bins=20, alpha=0.6, label='Deceptive', color='red')

    ax.axvline(honest_proj.mean(), color='green', linestyle='--', linewidth=2)
    ax.axvline(deceptive_proj.mean(), color='red', linestyle='--', linewidth=2)

    ax.set_xlabel('Projection onto Deception Direction')
    ax.set_ylabel('Count')
    ax.set_title(f'Layer {best_layer}: Honest vs Deceptive Projections')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / f"projection_distribution_layer_{best_layer}.png",
                dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved projection plot to {output_dir}")


def save_deception_direction(
    results: Dict[int, PairedAnalysisResult],
    best_layer: int,
    output_dir: Path,
):
    """Save deception directions for ALL layers (for flexible steering)."""
    # Save direction for each layer
    directions_dir = output_dir / "directions"
    directions_dir.mkdir(parents=True, exist_ok=True)

    for layer, result in results.items():
        np.save(directions_dir / f"deception_direction_layer_{layer:02d}.npy",
                result.mean_deception_direction)

    # Also save combined file for convenience
    all_directions = np.stack([
        results[l].mean_deception_direction
        for l in sorted(results.keys())
    ])
    np.save(output_dir / "deception_directions_all.npy", all_directions)

    # Save summary
    summary = {
        str(layer): {
            "consistency": float(r.direction_consistency),
            "cross_category": float(r.cross_category_consistency),
            "separability": float(r.projection_separability),
        }
        for layer, r in results.items()
    }
    summary["best_layer_generalization"] = best_layer
    summary["best_layer_separability"] = int(max(
        results.keys(), key=lambda l: results[l].projection_separability
    ))

    with open(output_dir / "paired_analysis_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved deception directions for all {len(results)} layers to {directions_dir}")


def main():
    config = get_config()

    # Run analysis
    results, honest_acts, deceptive_acts, diffs, categories = run_paired_analysis(config)

    # Find best layer (highest cross-category generalization)
    best_layer = max(results.keys(),
                     key=lambda l: results[l].cross_category_consistency)

    print(f"\n{'='*60}")
    print(f"BEST LAYER FOR GENERALIZATION: {best_layer}")
    print(f"  Consistency: {results[best_layer].direction_consistency:.3f}")
    print(f"  Cross-category: {results[best_layer].cross_category_consistency:.3f}")
    print(f"  Separability: {results[best_layer].projection_separability:.1f}")
    print(f"{'='*60}")

    # Generate plots
    figures_dir = config.results_dir / "figures"
    plot_results(results, figures_dir)
    plot_projection_distribution(
        honest_acts, deceptive_acts, results, best_layer, figures_dir
    )

    # Save direction
    save_deception_direction(results, best_layer, config.probes_dir)

    print("\nDone! Key findings:")
    print("  - If cross-category consistency > 0: direction generalizes!")
    print("  - If separability > 2: meaningful separation between conditions")
    print("\nNext: python steering.py (uses the new paired direction)")


if __name__ == "__main__":
    main()
