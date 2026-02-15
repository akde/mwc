#!/usr/bin/env python3
"""
Generate publication-ready thermal classification comparison figures.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# Set publication style
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})

# Data from experiments (ordered by F1 Macro)
methods_full = [
    ('InceptionTime', 0.521, 641000, 'ensemble'),
    ('TCN+Aug', 0.549, 51000, 'augmented'),
    ('TCN', 0.540, 51000, 'deep_learning'),
    ('MiniRocket', 0.537, 10000, 'feature_based'),
    ('BiLSTM', 0.535, 531000, 'deep_learning'),
    ('1D CNN', 0.527, 59000, 'deep_learning'),
    ('InceptionTime', 0.501, 647000, 'deep_learning'),
    ('BiGRU', 0.490, 133000, 'deep_learning'),
    ('TS2Vec', 0.423, 329000, 'self_supervised'),
    ('SVM', 0.383, 0, 'baseline'),
    ('Transformer', 0.368, 229000, 'deep_learning'),
    ('LITE', 0.290, 13000, 'failed'),
    ('Distillation', 0.208, 16000, 'failed'),
]

# Color scheme by category
colors_map = {
    'ensemble': '#2ecc71',      # Green for best
    'augmented': '#27ae60',     # Darker green
    'deep_learning': '#3498db', # Blue
    'feature_based': '#9b59b6', # Purple
    'self_supervised': '#f39c12', # Orange
    'baseline': '#7f8c8d',      # Gray
    'failed': '#e74c3c',        # Red for failed
}

# Figure 1: F1 Macro Rankings
fig1, ax1 = plt.subplots(figsize=(8, 5))

methods = [m[0] for m in methods_full]
f1_scores = [m[1] for m in methods_full]
categories = [m[3] for m in methods_full]
colors = [colors_map[c] for c in categories]

bars = ax1.barh(range(len(methods)), f1_scores, color=colors, edgecolor='black', linewidth=0.5)

# Add value labels
for i, (bar, score) in enumerate(zip(bars, f1_scores)):
    ax1.text(score + 0.008, i, f'{score:.3f}', va='center', ha='left', fontsize=9)

# Add SVM baseline line
svm_f1 = 0.383
ax1.axvline(x=svm_f1, color='#7f8c8d', linestyle='--', linewidth=1.5, label='SVM Baseline')

ax1.set_yticks(range(len(methods)))
ax1.set_yticklabels(methods)
ax1.set_xlabel('F1 Macro Score')
ax1.set_xlim(0, 0.65)
ax1.invert_yaxis()

# Legend
legend_patches = [
    mpatches.Patch(color='#2ecc71', label='Ensemble (Best)'),
    mpatches.Patch(color='#3498db', label='Deep Learning'),
    mpatches.Patch(color='#9b59b6', label='Feature-based'),
    mpatches.Patch(color='#f39c12', label='Self-supervised'),
    mpatches.Patch(color='#7f8c8d', label='Baseline'),
    mpatches.Patch(color='#e74c3c', label='Failed (<SVM)'),
]
ax1.legend(handles=legend_patches, loc='lower right', framealpha=0.9)

ax1.set_title('Thermal Classification: F1 Macro by Method')
plt.tight_layout()
plt.savefig('images/thermal_f1_rankings.pdf', format='pdf')
plt.savefig('images/thermal_f1_rankings.png', format='png')
print("Saved: images/thermal_f1_rankings.pdf")
plt.close()


# Figure 2: Ablation Improvement (delta from baseline)
fig2, ax2 = plt.subplots(figsize=(8, 5))

# Ablation data: (method, baseline_f1, optimized_f1, num_experiments)
ablation_data = [
    ('TCN', 0.248, 0.540, 13),        # +29.2 pp
    ('BiLSTM', 0.405, 0.535, 11),     # +13.0 pp
    ('InceptionTime', 0.373, 0.501, 9), # +12.8 pp
    ('BiGRU', 0.380, 0.490, 9),       # +11.0 pp
    ('1D CNN', 0.404, 0.527, 21),     # +12.3 pp
    ('Transformer', 0.289, 0.368, 18), # +7.9 pp
]

methods_abl = [d[0] for d in ablation_data]
baseline = [d[1] for d in ablation_data]
optimized = [d[2] for d in ablation_data]
improvements = [d[2] - d[1] for d in ablation_data]

x = np.arange(len(methods_abl))
width = 0.35

bars1 = ax2.bar(x - width/2, baseline, width, label='Before Ablation', color='#bdc3c7', edgecolor='black', linewidth=0.5)
bars2 = ax2.bar(x + width/2, optimized, width, label='After Ablation', color='#3498db', edgecolor='black', linewidth=0.5)

# Add improvement annotations
for i, (xi, imp) in enumerate(zip(x, improvements)):
    ax2.annotate(f'+{imp*100:.1f}pp',
                 xy=(xi + width/2, optimized[i] + 0.01),
                 ha='center', va='bottom', fontsize=8, fontweight='bold', color='#27ae60')

ax2.set_xlabel('Architecture')
ax2.set_ylabel('F1 Macro Score')
ax2.set_title('Impact of Architecture-Specific Ablation Studies')
ax2.set_xticks(x)
ax2.set_xticklabels(methods_abl, rotation=15, ha='right')
ax2.set_ylim(0, 0.65)
ax2.legend(loc='upper right')

# Add SVM baseline
ax2.axhline(y=0.383, color='#7f8c8d', linestyle='--', linewidth=1.5, label='SVM Baseline')
ax2.text(len(methods_abl)-0.5, 0.39, 'SVM Baseline', fontsize=8, color='#7f8c8d')

plt.tight_layout()
plt.savefig('images/thermal_ablation_improvement.pdf', format='pdf')
plt.savefig('images/thermal_ablation_improvement.png', format='png')
print("Saved: images/thermal_ablation_improvement.pdf")
plt.close()


# Figure 3: Per-class F1 heatmap for top methods
fig3, ax3 = plt.subplots(figsize=(7, 5))

# Per-class F1 data from test_metrics.json files
per_class_data = {
    'InceptionTime': {'glass': 0.757, 'metal': 0.485, 'paper': 0.273, 'plastic': 0.715},
    'TCN+Aug': {'glass': 0.686, 'metal': 0.644, 'paper': 0.211, 'plastic': 0.654},
    'MiniRocket': {'glass': 0.623, 'metal': 0.597, 'paper': 0.222, 'plastic': 0.706},
    'BiLSTM': {'glass': 0.762, 'metal': 0.222, 'paper': 0.381, 'plastic': 0.656},
    '1D CNN': {'glass': 0.722, 'metal': 0.394, 'paper': 0.303, 'plastic': 0.690},
    'SVM': {'glass': 0.576, 'metal': 0.277, 'paper': 0.111, 'plastic': 0.566},
}

methods_hm = list(per_class_data.keys())
classes = ['glass', 'metal', 'paper', 'plastic']
data_matrix = np.array([[per_class_data[m][c] for c in classes] for m in methods_hm])

im = ax3.imshow(data_matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Add text annotations
for i in range(len(methods_hm)):
    for j in range(len(classes)):
        val = data_matrix[i, j]
        color = 'white' if val < 0.35 or val > 0.65 else 'black'
        ax3.text(j, i, f'{val:.2f}', ha='center', va='center', color=color, fontsize=10, fontweight='bold')

ax3.set_xticks(range(len(classes)))
ax3.set_xticklabels([c.capitalize() for c in classes])
ax3.set_yticks(range(len(methods_hm)))
ax3.set_yticklabels(methods_hm)
ax3.set_xlabel('Material Class')
ax3.set_ylabel('Method')
ax3.set_title('Per-Class F1 Score by Method')

cbar = plt.colorbar(im, ax=ax3, shrink=0.8)
cbar.set_label('F1 Score')

plt.tight_layout()
plt.savefig('images/thermal_per_class_heatmap.pdf', format='pdf')
plt.savefig('images/thermal_per_class_heatmap.png', format='png')
print("Saved: images/thermal_per_class_heatmap.pdf")
plt.close()

print("\nAll figures generated successfully!")
