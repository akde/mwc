"""
Evaluation utilities for thermal time series classification.

This module provides:
- compute_metrics: Calculate accuracy, F1 scores, per-class metrics
- compute_confusion_matrix: Generate confusion matrix
- mcnemar_test: Statistical comparison between two classifiers
- pairwise_comparisons: All pairwise McNemar's tests with Bonferroni correction
- bootstrap_ci: Bootstrap confidence intervals
- plot_confusion_matrix: Visualization
- metrics_to_latex: LaTeX table generation
- save_predictions: Save predictions in standardized CSV format (RGB-aligned)
- save_classification_report: Save sklearn classification report
- save_metrics_json: Save metrics to JSON file
"""

import json
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)
from scipy.stats import chi2
from typing import Dict, Tuple, List, Optional, Any
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

from .config import CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_COMPARISONS, BONFERRONI_ALPHA, NUM_CLASSES


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = CLASS_NAMES
) -> Dict[str, Any]:
    """
    Compute comprehensive classification metrics.

    Key naming convention ALIGNED WITH RGB classifiers for fair comparison:
    - f1_weighted (not weighted_f1)
    - f1_macro (not macro_f1)

    Args:
        y_true: True labels (integer class indices)
        y_pred: Predicted labels (integer class indices)
        class_names: List of class names

    Returns:
        Dict with accuracy, F1 scores, precision, recall, per-class metrics, and support
    """
    # Handle empty arrays - return zero metrics instead of crashing
    if len(y_true) == 0 or len(y_pred) == 0:
        return {
            'accuracy': 0.0,
            'f1_weighted': 0.0,
            'f1_macro': 0.0,
            'precision_weighted': 0.0,
            'recall_weighted': 0.0,
            'per_class': {name: {'precision': 0.0, 'recall': 0.0, 'f1': 0.0, 'support': 0} for name in class_names},
            'support': {name: 0 for name in class_names}
        }

    # Compute overall metrics (key names ALIGNED WITH RGB)
    metrics = {
        'accuracy': float(accuracy_score(y_true, y_pred)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro', zero_division=0)),
        'precision_weighted': float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall_weighted': float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'per_class': {}
    }

    # Per-class metrics - use explicit labels for consistent array size
    all_labels = list(range(NUM_CLASSES))
    precision_per_class = precision_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, labels=all_labels, zero_division=0)

    # Support (sample counts) - build explicitly to handle missing classes
    unique_labels, counts = np.unique(y_true, return_counts=True)
    count_dict = dict(zip(unique_labels, counts))
    support = {name: int(count_dict.get(CLASS_TO_IDX[name], 0)) for name in class_names}

    for i, name in enumerate(class_names):
        metrics['per_class'][name] = {
            'precision': float(precision_per_class[i]),
            'recall': float(recall_per_class[i]),
            'f1': float(f1_per_class[i]),
            'support': int(support.get(name, 0))
        }

    metrics['support'] = support

    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    normalize: bool = False
) -> np.ndarray:
    """
    Compute confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        normalize: Normalize rows to sum to 1

    Returns:
        Confusion matrix array [n_classes, n_classes]
    """
    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Handle division by zero
    return cm


def mcnemar_test(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    correction: bool = True
) -> Tuple[float, float]:
    """
    McNemar's test for comparing two classifiers on the same test set.

    Null hypothesis: Both classifiers have the same error rate.

    The test looks at discordant pairs:
    - n12: samples where A is correct, B is wrong
    - n21: samples where A is wrong, B is correct

    Args:
        y_true: True labels
        y_pred_a: Predictions from classifier A
        y_pred_b: Predictions from classifier B
        correction: Apply continuity correction (recommended for small samples)

    Returns:
        Tuple of (chi2 statistic, p-value)
    """
    correct_a = (y_pred_a == y_true)
    correct_b = (y_pred_b == y_true)

    # Count discordant pairs
    n12 = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    n21 = np.sum(~correct_a & correct_b)  # A wrong, B correct

    # Handle edge case
    if n12 + n21 == 0:
        return 0.0, 1.0  # No difference

    # McNemar's test statistic with continuity correction
    if correction:
        statistic = (abs(n12 - n21) - 1) ** 2 / (n12 + n21)
    else:
        statistic = (n12 - n21) ** 2 / (n12 + n21)

    # Chi-squared test with 1 degree of freedom
    p_value = 1 - chi2.cdf(statistic, df=1)

    return statistic, p_value


def pairwise_comparisons(
    predictions: Dict[str, np.ndarray],
    y_true: np.ndarray,
    alpha: float = 0.05
) -> pd.DataFrame:
    """
    Perform all pairwise McNemar's tests with Bonferroni correction.

    Args:
        predictions: Dict mapping model_name -> predictions array
        y_true: True labels
        alpha: Significance level before correction

    Returns:
        DataFrame with columns: model_a, model_b, statistic, p_value, significant
    """
    model_names = list(predictions.keys())
    n_comparisons = len(list(combinations(model_names, 2)))
    bonferroni_alpha = alpha / n_comparisons

    results = []
    for model_a, model_b in combinations(model_names, 2):
        stat, p_value = mcnemar_test(y_true, predictions[model_a], predictions[model_b])
        results.append({
            'model_a': model_a,
            'model_b': model_b,
            'statistic': stat,
            'p_value': p_value,
            'significant': p_value < bonferroni_alpha
        })

    df = pd.DataFrame(results)
    df['bonferroni_alpha'] = bonferroni_alpha
    return df


def bootstrap_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = 1000,
    ci: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        metric_fn: Function that takes (y_true, y_pred) and returns a score
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval (e.g., 0.95 for 95% CI)
        random_state: Random seed

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper)
    """
    np.random.seed(random_state)
    n = len(y_true)
    scores = []

    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        score = metric_fn(y_true[idx], y_pred[idx])
        scores.append(score)

    point_estimate = metric_fn(y_true, y_pred)
    ci_lower = np.percentile(scores, (1 - ci) / 2 * 100)
    ci_upper = np.percentile(scores, (1 + ci) / 2 * 100)

    return point_estimate, ci_lower, ci_upper


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str] = CLASS_NAMES,
    title: str = 'Confusion Matrix',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6),
    normalize: bool = False,
    show_counts: bool = True,
    cmap: str = 'Blues'
):
    """
    Plot confusion matrix with annotations.

    Args:
        cm: Confusion matrix array
        class_names: Class labels
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
        normalize: Show percentages instead of counts
        show_counts: Show counts in cells even if normalized
        cmap: Color map
    """
    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm_normalized = np.nan_to_num(cm_normalized)
        im = ax.imshow(cm_normalized, interpolation='nearest', cmap=cmap)
    else:
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)

    # Colorbar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel('Percentage' if normalize else 'Count', rotation=-90, va="bottom")

    # Ticks
    ax.set(xticks=np.arange(len(class_names)),
           yticks=np.arange(len(class_names)),
           xticklabels=class_names,
           yticklabels=class_names,
           title=title,
           ylabel='True Label',
           xlabel='Predicted Label')

    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Annotations
    thresh = (cm.max() + cm.min()) / 2 if not normalize else 0.5
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if normalize:
                if show_counts:
                    text = f'{cm_normalized[i, j]:.1%}\n({cm[i, j]})'
                else:
                    text = f'{cm_normalized[i, j]:.1%}'
                color = 'white' if cm_normalized[i, j] > thresh else 'black'
            else:
                text = str(cm[i, j])
                color = 'white' if cm[i, j] > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_learning_curves(
    history: Dict,
    title: str = 'Training History',
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 4)
):
    """
    Plot training and validation loss/accuracy curves.

    Args:
        history: Dict with 'train_loss', 'val_loss', 'train_acc', 'val_acc'
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train')
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Validation')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title} - Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, [a * 100 for a in history['train_acc']], 'b-', label='Train')
    axes[1].plot(epochs, [a * 100 for a in history['val_acc']], 'r-', label='Validation')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title(f'{title} - Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def metrics_to_latex(
    results: Dict[str, Dict],
    caption: str = 'Classification Performance Comparison',
    label: str = 'tab:comparison'
) -> str:
    """
    Generate LaTeX table from model comparison results.

    Args:
        results: Dict mapping model_name -> metrics dict
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(f'\\caption{{{caption}}}')
    latex.append(f'\\label{{{label}}}')
    latex.append(r'\begin{tabular}{lccc}')
    latex.append(r'\toprule')
    latex.append(r'Method & Accuracy & Weighted F1 & Macro F1 \\')
    latex.append(r'\midrule')

    # Find best values (support both old and new key names for backward compatibility)
    def get_f1_weighted(r):
        return r.get('f1_weighted', r.get('weighted_f1', 0))

    def get_f1_macro(r):
        return r.get('f1_macro', r.get('macro_f1', 0))

    best_acc = max(r['accuracy'] for r in results.values())
    best_wf1 = max(get_f1_weighted(r) for r in results.values())
    best_mf1 = max(get_f1_macro(r) for r in results.values())

    for name, metrics in results.items():
        acc = metrics['accuracy']
        wf1 = get_f1_weighted(metrics)
        mf1 = get_f1_macro(metrics)

        # Bold best values
        acc_str = f"\\textbf{{{acc*100:.1f}\\%}}" if acc == best_acc else f"{acc*100:.1f}\\%"
        wf1_str = f"\\textbf{{{wf1:.3f}}}" if wf1 == best_wf1 else f"{wf1:.3f}"
        mf1_str = f"\\textbf{{{mf1:.3f}}}" if mf1 == best_mf1 else f"{mf1:.3f}"

        latex.append(f'{name} & {acc_str} & {wf1_str} & {mf1_str} \\\\')

    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


def per_class_to_latex(
    metrics: Dict,
    model_name: str,
    caption: str = None,
    label: str = None
) -> str:
    """
    Generate LaTeX table for per-class metrics.

    Args:
        metrics: Metrics dict with 'per_class' key
        model_name: Name of the model
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    if caption is None:
        caption = f'Per-Class Classification Performance ({model_name})'
    if label is None:
        label = f'tab:{model_name.lower()}_per_class'

    # Support both old and new key names for backward compatibility
    f1_weighted = metrics.get('f1_weighted', metrics.get('weighted_f1', 0))

    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(f'\\caption{{{caption}}}')
    latex.append(f'\\label{{{label}}}')
    latex.append(r'\begin{tabular}{lcccc}')
    latex.append(r'\toprule')
    latex.append(r'Class & Precision & Recall & F1-Score & Support \\')
    latex.append(r'\midrule')

    for class_name in CLASS_NAMES:
        if class_name in metrics['per_class']:
            pc = metrics['per_class'][class_name]
            latex.append(
                f"{class_name.capitalize()} & {pc['precision']:.3f} & "
                f"{pc['recall']:.3f} & {pc['f1']:.3f} & {pc['support']} \\\\"
            )

    latex.append(r'\midrule')
    latex.append(
        f"Weighted Avg & - & - & {f1_weighted:.3f} & "
        f"{sum(metrics['per_class'][c]['support'] for c in CLASS_NAMES)} \\\\"
    )
    latex.append(r'\bottomrule')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


def significance_matrix_to_latex(
    comparisons_df: pd.DataFrame,
    model_names: List[str],
    caption: str = 'Pairwise Statistical Significance (McNemar Test)',
    label: str = 'tab:significance'
) -> str:
    """
    Generate LaTeX table showing p-values for all pairwise comparisons.

    Args:
        comparisons_df: DataFrame from pairwise_comparisons()
        model_names: List of model names (in order)
        caption: Table caption
        label: Table label

    Returns:
        LaTeX table string
    """
    n = len(model_names)

    # Build p-value matrix
    pvalue_matrix = np.ones((n, n))
    for _, row in comparisons_df.iterrows():
        i = model_names.index(row['model_a'])
        j = model_names.index(row['model_b'])
        pvalue_matrix[i, j] = row['p_value']
        pvalue_matrix[j, i] = row['p_value']

    latex = []
    latex.append(r'\begin{table}[htbp]')
    latex.append(r'\centering')
    latex.append(f'\\caption{{{caption}}}')
    latex.append(f'\\label{{{label}}}')
    latex.append(r'\begin{tabular}{l' + 'c' * n + '}')
    latex.append(r'\toprule')
    latex.append(' & ' + ' & '.join(model_names) + r' \\')
    latex.append(r'\midrule')

    bonferroni_alpha = comparisons_df['bonferroni_alpha'].iloc[0]

    for i, name_i in enumerate(model_names):
        row_values = []
        for j, name_j in enumerate(model_names):
            if i == j:
                row_values.append('-')
            elif i < j:
                p = pvalue_matrix[i, j]
                if p < bonferroni_alpha:
                    row_values.append(f'\\textbf{{{p:.4f}}}')
                else:
                    row_values.append(f'{p:.4f}')
            else:
                row_values.append('')

        latex.append(f'{name_i} & ' + ' & '.join(row_values) + r' \\')

    latex.append(r'\bottomrule')
    latex.append(r'\multicolumn{' + str(n+1) + r'}{l}{\small Bonferroni-corrected $\alpha$ = ' +
                 f'{bonferroni_alpha:.4f}' + r'} \\')
    latex.append(r'\multicolumn{' + str(n+1) + r'}{l}{\small Bold: Statistically significant difference} \\')
    latex.append(r'\end{tabular}')
    latex.append(r'\end{table}')

    return '\n'.join(latex)


def save_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    unique_ids: List[str],
    output_path: Path,
    probabilities: np.ndarray = None,
    class_names: List[str] = None
) -> None:
    """
    Save predictions to CSV file in standardized format (ALIGNED WITH RGB).

    Output format matches RGB classifier predictions for cross-domain comparison:
    unique_id, ground_truth, predicted, correct, prob_metal, prob_plastic, prob_glass, prob_paper

    Args:
        y_true: Ground truth labels (integer indices)
        y_pred: Predicted labels (integer indices)
        unique_ids: Unique identifiers for each sample
        output_path: Path to save CSV
        probabilities: Optional prediction probabilities [N, num_classes]
        class_names: Class names for probability columns
    """
    if class_names is None:
        class_names = CLASS_NAMES

    # Convert indices to class names
    y_true_names = [IDX_TO_CLASS[int(y)] for y in y_true]
    y_pred_names = [IDX_TO_CLASS[int(y)] for y in y_pred]

    # Create DataFrame (ALIGNED WITH RGB format)
    data = {
        'unique_id': unique_ids,
        'ground_truth': y_true_names,
        'predicted': y_pred_names,
        'correct': [gt == pred for gt, pred in zip(y_true_names, y_pred_names)]
    }

    # Add probabilities if provided
    if probabilities is not None:
        for idx, name in enumerate(class_names):
            data[f'prob_{name}'] = probabilities[:, idx]

    df = pd.DataFrame(data)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    class_names: List[str] = None
) -> str:
    """
    Save sklearn classification report to file.

    Args:
        y_true: Ground truth labels (integer indices)
        y_pred: Predicted labels (integer indices)
        output_path: Path to save report
        class_names: Class names for report

    Returns:
        Classification report string
    """
    if class_names is None:
        class_names = CLASS_NAMES

    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)

    return report


def save_metrics_json(
    metrics: Dict[str, Any],
    output_path: Path
) -> None:
    """
    Save metrics dictionary to JSON file.

    Args:
        metrics: Metrics dictionary
        output_path: Path to save JSON
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(metrics, f, indent=2)


if __name__ == '__main__':
    # Test with synthetic data
    np.random.seed(42)
    n_samples = 100

    y_true = np.random.randint(0, 4, n_samples)
    y_pred_a = y_true.copy()
    y_pred_a[np.random.choice(n_samples, 20, replace=False)] = np.random.randint(0, 4, 20)

    y_pred_b = y_true.copy()
    y_pred_b[np.random.choice(n_samples, 30, replace=False)] = np.random.randint(0, 4, 30)

    print("Testing evaluation metrics...")

    # Compute metrics
    metrics_a = compute_metrics(y_true, y_pred_a)
    print(f"\nModel A - Accuracy: {metrics_a['accuracy']:.3f}, F1 Weighted: {metrics_a['f1_weighted']:.3f}")

    metrics_b = compute_metrics(y_true, y_pred_b)
    print(f"Model B - Accuracy: {metrics_b['accuracy']:.3f}, F1 Weighted: {metrics_b['f1_weighted']:.3f}")

    # McNemar's test
    stat, p_value = mcnemar_test(y_true, y_pred_a, y_pred_b)
    print(f"\nMcNemar's test: statistic={stat:.3f}, p-value={p_value:.4f}")

    # Bootstrap CI
    point, lower, upper = bootstrap_ci(y_true, y_pred_a, accuracy_score)
    print(f"\nBootstrap 95% CI for Model A accuracy: {point:.3f} [{lower:.3f}, {upper:.3f}]")

    # LaTeX table
    print("\nLaTeX comparison table:")
    print(metrics_to_latex({'Model A': metrics_a, 'Model B': metrics_b}))

    print("\nAll tests passed!")
