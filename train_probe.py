"""
Train linear probes to detect deceptive intent from model activations.

This script:
1. Loads saved activations from collect_activations.py
2. Trains logistic regression probes for each layer
3. Evaluates probe accuracy and reports results
4. Saves trained probes for later use
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

from config import get_config, ExperimentConfig


@dataclass
class ProbeResult:
    """Results from training a probe on one layer"""
    layer: int
    accuracy: float
    auc_roc: float
    cv_scores: List[float]
    cv_mean: float
    cv_std: float
    coefficients: np.ndarray
    intercept: float


def load_activations(
    activations_dir: Path,
    layers: Optional[List[int]] = None
) -> Tuple[Dict[int, np.ndarray], np.ndarray, dict]:
    """
    Load saved activations from disk.

    Args:
        activations_dir: Directory containing activation files
        layers: Specific layers to load (None = all)

    Returns:
        Tuple of (activations dict, labels array, metadata dict)
    """
    # Load metadata
    with open(activations_dir / "metadata.json") as f:
        metadata = json.load(f)

    # Load labels
    labels = np.load(activations_dir / "labels.npy")

    # Determine which layers to load
    available_layers = metadata["layers"]
    if layers is None:
        layers_to_load = available_layers
    else:
        layers_to_load = [l for l in layers if l in available_layers]

    # Load activations
    activations = {}
    for layer in layers_to_load:
        filepath = activations_dir / f"activations_layer_{layer:02d}.npy"
        if filepath.exists():
            activations[layer] = np.load(filepath)
            print(f"Loaded layer {layer}: shape {activations[layer].shape}")

    return activations, labels, metadata


def train_probe(
    X: np.ndarray,
    y: np.ndarray,
    regularization: float = 1.0,
    random_state: int = 42,
    test_size: float = 0.2,
) -> Tuple[LogisticRegression, StandardScaler, ProbeResult]:
    """
    Train a logistic regression probe on activations.

    Args:
        X: Activation features (n_samples, hidden_size)
        y: Labels (0 = honest, 1 = deceptive)
        regularization: L2 regularization strength
        random_state: Random seed
        test_size: Fraction for test split

    Returns:
        Tuple of (trained model, scaler, results)
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train probe
    probe = LogisticRegression(
        C=1/regularization,
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    probe.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = probe.predict(X_test_scaled)
    y_prob = probe.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    auc_roc = roc_auc_score(y_test, y_prob)

    # Cross-validation on full dataset
    X_scaled_full = scaler.fit_transform(X)
    cv_scores = cross_val_score(
        LogisticRegression(C=1/regularization, max_iter=1000, solver='lbfgs'),
        X_scaled_full, y, cv=5, scoring='accuracy'
    )

    return probe, scaler, ProbeResult(
        layer=-1,  # Will be set by caller
        accuracy=accuracy,
        auc_roc=auc_roc,
        cv_scores=cv_scores.tolist(),
        cv_mean=cv_scores.mean(),
        cv_std=cv_scores.std(),
        coefficients=probe.coef_[0],
        intercept=probe.intercept_[0]
    )


def train_all_layers(
    activations: Dict[int, np.ndarray],
    labels: np.ndarray,
    config: ExperimentConfig
) -> Dict[int, Tuple[LogisticRegression, StandardScaler, ProbeResult]]:
    """
    Train probes for all layers.

    Args:
        activations: Dict of layer -> activation array
        labels: Label array
        config: Experiment configuration

    Returns:
        Dict of layer -> (probe, scaler, results)
    """
    results = {}

    for layer in sorted(activations.keys()):
        print(f"\nTraining probe for layer {layer}...")

        X = activations[layer]
        probe, scaler, result = train_probe(
            X, labels,
            regularization=config.probe.regularization,
            random_state=config.probe.random_state,
            test_size=config.probe.test_size
        )
        result.layer = layer

        print(f"  Accuracy: {result.accuracy:.3f}")
        print(f"  AUC-ROC: {result.auc_roc:.3f}")
        print(f"  CV Mean: {result.cv_mean:.3f} (+/- {result.cv_std:.3f})")

        results[layer] = (probe, scaler, result)

    return results


def save_probes(
    results: Dict[int, Tuple[LogisticRegression, StandardScaler, ProbeResult]],
    output_dir: Path
):
    """Save trained probes and results."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save each probe
    for layer, (probe, scaler, result) in results.items():
        probe_path = output_dir / f"probe_layer_{layer:02d}.pkl"
        with open(probe_path, 'wb') as f:
            pickle.dump({'probe': probe, 'scaler': scaler}, f)

    # Save summary results
    summary = {
        layer: {
            'layer': result.layer,
            'accuracy': result.accuracy,
            'auc_roc': result.auc_roc,
            'cv_mean': result.cv_mean,
            'cv_std': result.cv_std,
        }
        for layer, (_, _, result) in results.items()
    }

    with open(output_dir / "probe_results.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved probes to {output_dir}")


def find_best_layer(
    results: Dict[int, Tuple[LogisticRegression, StandardScaler, ProbeResult]]
) -> int:
    """Find the layer with the best probe accuracy."""
    best_layer = max(
        results.keys(),
        key=lambda l: results[l][2].cv_mean
    )
    return best_layer


def get_deception_direction(
    probe: LogisticRegression,
    scaler: StandardScaler
) -> np.ndarray:
    """
    Extract the deception direction vector from a trained probe.

    This is the direction in activation space that the probe uses
    to distinguish honest from deceptive activations.
    """
    # The coefficients point from honest toward deceptive
    # We need to account for the scaling
    direction = probe.coef_[0] / scaler.scale_
    # Normalize to unit vector
    direction = direction / np.linalg.norm(direction)
    return direction


def main():
    """Main function to train probes."""
    config = get_config()

    # Load activations
    print("Loading activations...")
    activations, labels, metadata = load_activations(config.activations_dir)

    print(f"\nLoaded {len(labels)} samples")
    print(f"Label distribution: {np.bincount(labels)}")
    print(f"Layers: {list(activations.keys())}")

    # Train probes
    results = train_all_layers(activations, labels, config)

    # Find best layer
    best_layer = find_best_layer(results)
    best_result = results[best_layer][2]

    print(f"\n{'='*50}")
    print(f"BEST LAYER: {best_layer}")
    print(f"  Accuracy: {best_result.accuracy:.3f}")
    print(f"  AUC-ROC: {best_result.auc_roc:.3f}")
    print(f"  CV Mean: {best_result.cv_mean:.3f} (+/- {best_result.cv_std:.3f})")
    print(f"{'='*50}")

    # Save probes
    save_probes(results, config.probes_dir)

    # Save deception direction for best layer
    probe, scaler, _ = results[best_layer]
    deception_direction = get_deception_direction(probe, scaler)
    np.save(config.probes_dir / "deception_direction.npy", deception_direction)
    print(f"Saved deception direction vector to {config.probes_dir / 'deception_direction.npy'}")

    print("\nDone! Next steps:")
    print("  - python analyze_layers.py  (visualize layer-wise results)")
    print("  - python steering.py        (test activation steering)")


if __name__ == "__main__":
    main()
