"""
Analyze and visualize probe performance across layers.

This script:
1. Loads probe results from training
2. Creates visualizations of layer-wise performance
3. Analyzes generalization across vulnerability categories
4. Generates publication-ready figures
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import pickle

from config import get_config, ExperimentConfig
from train_probe import load_activations


def load_probe_results(probes_dir: Path) -> Dict[int, dict]:
    """Load probe training results."""
    with open(probes_dir / "probe_results.json") as f:
        return json.load(f)


def plot_layer_accuracy(results: Dict[str, dict], output_path: Path):
    """
    Plot probe accuracy across layers.

    This is a key figure - shows where in the network deceptive intent
    is most detectable.
    """
    # Convert string keys to int and sort
    layers = sorted([int(k) for k in results.keys()])
    accuracies = [results[str(l)]['accuracy'] for l in layers]
    cv_means = [results[str(l)]['cv_mean'] for l in layers]
    cv_stds = [results[str(l)]['cv_std'] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot CV mean with error bars
    ax.errorbar(layers, cv_means, yerr=cv_stds, fmt='o-', capsize=5,
                label='Cross-validation accuracy', color='blue', linewidth=2)

    # Plot test accuracy
    ax.scatter(layers, accuracies, marker='x', s=100, color='red',
               label='Test accuracy', zorder=5)

    # Add baseline
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')

    # Highlight best layer
    best_layer = max(layers, key=lambda l: results[str(l)]['cv_mean'])
    ax.axvline(x=best_layer, color='green', linestyle=':', alpha=0.7,
               label=f'Best layer ({best_layer})')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Deception Probe Accuracy Across Layers', fontsize=14)
    ax.legend(loc='lower right')
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved layer accuracy plot to {output_path}")


def plot_auc_roc(results: Dict[str, dict], output_path: Path):
    """Plot AUC-ROC across layers."""
    layers = sorted([int(k) for k in results.keys()])
    auc_scores = [results[str(l)]['auc_roc'] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, auc_scores, 'o-', color='purple', linewidth=2, markersize=8)
    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')

    # Highlight best layer
    best_layer = max(layers, key=lambda l: results[str(l)]['auc_roc'])
    ax.axvline(x=best_layer, color='green', linestyle=':', alpha=0.7,
               label=f'Best layer ({best_layer})')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('AUC-ROC', fontsize=12)
    ax.set_title('Deception Probe AUC-ROC Across Layers', fontsize=14)
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved AUC-ROC plot to {output_path}")


def analyze_category_generalization(
    activations_dir: Path,
    probes_dir: Path,
    config: ExperimentConfig,
    output_path: Path
):
    """
    Test if probes trained on some categories generalize to others.

    This is crucial for the safety application - we want a detector
    that works on NEW types of deception, not just the ones we trained on.
    """
    # Load activations and metadata
    activations, labels, metadata = load_activations(activations_dir)
    categories = np.array(metadata['categories'])

    # Get train and test categories from config
    train_cats = set(config.data.train_categories)
    test_cats = set(config.data.test_categories)

    # Split data by category
    train_mask = np.array([c in train_cats for c in categories])
    test_mask = np.array([c in test_cats for c in categories])

    print(f"\nGeneralization Analysis:")
    print(f"  Training categories: {train_cats}")
    print(f"  Testing categories: {test_cats}")
    print(f"  Train samples: {train_mask.sum()}")
    print(f"  Test samples: {test_mask.sum()}")

    results = {}

    for layer in sorted(activations.keys()):
        X = activations[layer]

        # Train on train categories
        X_train = X[train_mask]
        y_train = labels[train_mask]

        # Test on held-out categories
        X_test = X[test_mask]
        y_test = labels[test_mask]

        # Train probe
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        probe = LogisticRegression(max_iter=1000, solver='lbfgs')
        probe.fit(X_train_scaled, y_train)

        # Evaluate
        train_acc = probe.score(X_train_scaled, y_train)
        test_acc = probe.score(X_test_scaled, y_test)

        results[layer] = {
            'train_acc': train_acc,
            'test_acc': test_acc,
            'generalization_gap': train_acc - test_acc
        }

        print(f"Layer {layer}: train={train_acc:.3f}, test={test_acc:.3f}, gap={train_acc - test_acc:.3f}")

    # Plot
    layers = sorted(results.keys())
    train_accs = [results[l]['train_acc'] for l in layers]
    test_accs = [results[l]['test_acc'] for l in layers]

    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(layers, train_accs, 'o-', color='blue', linewidth=2, label='Train categories')
    ax.plot(layers, test_accs, 's-', color='orange', linewidth=2, label='Held-out categories')
    ax.fill_between(layers, test_accs, train_accs, alpha=0.2, color='gray')

    ax.axhline(y=0.5, color='gray', linestyle='--', label='Random baseline')

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Generalization: Train vs Held-out Vulnerability Categories', fontsize=14)
    ax.legend()
    ax.set_ylim(0.4, 1.0)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"\nSaved generalization plot to {output_path}")

    # Save results
    with open(probes_dir / "generalization_results.json", 'w') as f:
        json.dump({str(k): v for k, v in results.items()}, f, indent=2)

    return results


def plot_activation_pca(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    categories: np.ndarray,
    layer: int,
    output_path: Path
):
    """
    PCA visualization of activations colored by honest/deceptive.

    Shows if honest and deceptive activations cluster separately.
    """
    from sklearn.decomposition import PCA

    X = activations[layer]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))

    colors = ['#2ecc71', '#e74c3c']  # green for honest, red for deceptive
    labels_text = ['Honest', 'Deceptive']

    for label in [0, 1]:
        mask = labels == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                  c=colors[label], label=labels_text[label],
                  alpha=0.6, s=50)

    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)', fontsize=12)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)', fontsize=12)
    ax.set_title(f'Activation PCA (Layer {layer})', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"Saved PCA plot to {output_path}")


def create_summary_table(results: Dict[str, dict], output_path: Path):
    """Create a summary table of results."""
    layers = sorted([int(k) for k in results.keys()])

    lines = [
        "# Probe Results Summary\n",
        "| Layer | Accuracy | AUC-ROC | CV Mean | CV Std |",
        "|-------|----------|---------|---------|--------|"
    ]

    for layer in layers:
        r = results[str(layer)]
        lines.append(
            f"| {layer:5d} | {r['accuracy']:.3f}    | {r['auc_roc']:.3f}   | "
            f"{r['cv_mean']:.3f}   | {r['cv_std']:.3f}  |"
        )

    # Find best layer
    best_layer = max(layers, key=lambda l: results[str(l)]['cv_mean'])
    lines.append(f"\n**Best Layer: {best_layer}**")

    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))

    print(f"Saved summary table to {output_path}")


def main():
    """Main analysis function."""
    config = get_config()

    # Load results
    print("Loading probe results...")
    results = load_probe_results(config.probes_dir)

    # Create output directory
    figures_dir = config.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Generate plots
    print("\nGenerating visualizations...")

    plot_layer_accuracy(results, figures_dir / "layer_accuracy.png")
    plot_auc_roc(results, figures_dir / "layer_auc_roc.png")

    # Load activations for additional analyses
    activations, labels, metadata = load_activations(config.activations_dir)

    # Generalization analysis
    print("\nRunning generalization analysis...")
    analyze_category_generalization(
        config.activations_dir,
        config.probes_dir,
        config,
        figures_dir / "generalization.png"
    )

    # PCA for best layer
    best_layer = max(results.keys(), key=lambda l: results[l]['cv_mean'])
    best_layer_int = int(best_layer)
    if best_layer_int in activations:
        plot_activation_pca(
            activations, labels, np.array(metadata['categories']),
            best_layer_int,
            figures_dir / f"pca_layer_{best_layer_int}.png"
        )

    # Summary table
    create_summary_table(results, config.results_dir / "results_summary.md")

    print("\nAnalysis complete!")
    print(f"Figures saved to: {figures_dir}")
    print(f"Summary saved to: {config.results_dir / 'results_summary.md'}")


if __name__ == "__main__":
    main()
