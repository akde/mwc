#!/usr/bin/env python3
"""
Generate publication-quality figures for thermal classification results.

Creates two-panel figure:
(a) Final F1 Macro rankings
(b) Ablation study improvement (before vs after)

Output: PDF format for LaTeX inclusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path

# Publication settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 11,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.05,
})

# Verified data from test_metrics.json files
FINAL_RESULTS = {
    'TCN': 0.5396,
    'MiniRocket': 0.5370,
    'BiLSTM': 0.5347,
    'InceptionTime': 0.5008,
    'BiGRU': 0.4898,
    'SVM': 0.3826,
    'Transformer': 0.3684,
}

ABLATION_DATA = {
    # Model: (baseline, optimized, experiments)
    'TCN': (0.2478, 0.5396, 13),
    'BiLSTM': (0.4049, 0.5347, 11),
    'InceptionTime': (0.3732, 0.5008, 9),
    'BiGRU': (0.4409, 0.4898, 9),
    'Transformer': (0.2898, 0.3684, 18),
}

# Colors - colorblind-friendly palette
COLORS = {
    'TCN': '#2166ac',           # Blue
    'MiniRocket': '#67a9cf',    # Light blue
    'BiLSTM': '#1a9850',        # Green
    'InceptionTime': '#91cf60', # Light green
    'BiGRU': '#fee08b',         # Yellow
    'SVM': '#d73027',           # Red
    'Transformer': '#f46d43',   # Orange
    'baseline': '#bdbdbd',      # Gray for baseline
}


def create_rankings_figure(ax):
    """Create horizontal bar chart of final F1 Macro rankings."""
    models = list(FINAL_RESULTS.keys())
    f1_scores = list(FINAL_RESULTS.values())
    colors = [COLORS[m] for m in models]

    y_pos = np.arange(len(models))

    bars = ax.barh(y_pos, f1_scores, color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, score) in enumerate(zip(bars, f1_scores)):
        ax.text(score + 0.01, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', ha='left', fontsize=9)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(models)
    ax.set_xlabel('F1 Macro Score')
    ax.set_xlim(0, 0.65)
    ax.axvline(x=FINAL_RESULTS['SVM'], color='red', linestyle='--',
               linewidth=1, alpha=0.7, label='SVM baseline')
    ax.invert_yaxis()  # Best model at top
    ax.set_title('(a) Final Classification Performance', fontweight='bold', pad=10)

    # Grid
    ax.set_axisbelow(True)
    ax.xaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def create_ablation_figure(ax):
    """Create grouped bar chart showing before/after ablation improvement."""
    models = list(ABLATION_DATA.keys())
    baselines = [ABLATION_DATA[m][0] for m in models]
    optimized = [ABLATION_DATA[m][1] for m in models]
    improvements = [opt - base for base, opt in zip(baselines, optimized)]

    x = np.arange(len(models))
    width = 0.35

    # Bars
    bars1 = ax.bar(x - width/2, baselines, width, label='Baseline',
                   color=COLORS['baseline'], edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x + width/2, optimized, width, label='Optimized',
                   color=[COLORS[m] for m in models], edgecolor='black', linewidth=0.5)

    # Add improvement annotations
    for i, (bar, imp) in enumerate(zip(bars2, improvements)):
        ax.annotate(f'+{imp*100:.1f}pp',
                   xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                   xytext=(0, 3), textcoords='offset points',
                   ha='center', va='bottom', fontsize=8, fontweight='bold',
                   color='#1a1a1a')

    ax.set_ylabel('F1 Macro Score')
    ax.set_xlabel('Model')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha='right')
    ax.set_ylim(0, 0.65)
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_title('(b) Ablation Study Improvement', fontweight='bold', pad=10)

    # Grid
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def main():
    """Generate and save the two-panel figure."""
    output_dir = Path(__file__).parent.parent / 'images'
    output_dir.mkdir(exist_ok=True)

    # Create combined figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    create_rankings_figure(ax1)
    create_ablation_figure(ax2)

    plt.tight_layout()

    # Save combined figure
    combined_path = output_dir / 'thermal_classification_results.pdf'
    fig.savefig(combined_path, format='pdf', bbox_inches='tight')
    print(f"Saved: {combined_path}")

    # Also save individual subfigures for flexibility
    fig1, ax1_single = plt.subplots(figsize=(5, 4))
    create_rankings_figure(ax1_single)
    fig1.tight_layout()
    fig1.savefig(output_dir / 'thermal_f1_rankings.pdf', format='pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'thermal_f1_rankings.pdf'}")
    plt.close(fig1)

    fig2, ax2_single = plt.subplots(figsize=(5, 4))
    create_ablation_figure(ax2_single)
    fig2.tight_layout()
    fig2.savefig(output_dir / 'thermal_ablation_improvement.pdf', format='pdf', bbox_inches='tight')
    print(f"Saved: {output_dir / 'thermal_ablation_improvement.pdf'}")
    plt.close(fig2)

    plt.close(fig)
    print("\nAll figures generated successfully!")


if __name__ == '__main__':
    main()
