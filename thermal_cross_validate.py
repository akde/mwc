"""5-fold stratified cross-validation for thermal material classifier."""

import numpy as np
import yaml
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix

from material_classifier.data.dataset import CLASS_NAMES
from cross_validate import load_labels, write_labels, train_one_fold, evaluate_fold


def main():
    config_path = "material_classifier/config/thermal.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    labels_csv = config["data"]["labels_csv"]
    rows = load_labels(labels_csv)

    # Extract classes for stratification
    classes = [r["class"] for r in rows]
    class_to_idx = {c: i for i, c in enumerate(CLASS_NAMES)}
    y = np.array([class_to_idx[c] for c in classes])

    n_folds = 5
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    all_preds = []
    all_labels = []
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(rows, y)):
        print(f"\n{'='*60}")
        print(f"Fold {fold_idx + 1}/{n_folds}  (train={len(train_idx)}, val={len(val_idx)})")
        print(f"{'='*60}")

        # Write fold splits to labels.csv
        fold_rows = []
        for i, r in enumerate(rows):
            row = dict(r)
            row["split"] = "train" if i in set(train_idx) else "val"
            fold_rows.append(row)
        write_labels(labels_csv, fold_rows)

        # Train
        best_state, best_f1 = train_one_fold(config, fold_idx)
        print(f"  Best val F1 during training: {best_f1:.4f}")

        # Evaluate
        preds, labels = evaluate_fold(config, best_state)
        all_preds.extend(preds)
        all_labels.extend(labels)

        # Per-fold metrics
        acc = accuracy_score(labels, preds)
        f1 = f1_score(labels, preds, average="macro", zero_division=0)
        fold_results.append({"fold": fold_idx + 1, "accuracy": acc, "macro_f1": f1, "n": len(labels)})
        print(f"  Fold {fold_idx + 1} — Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")

    # Aggregate results
    print(f"\n{'='*60}")
    print(f"Thermal 5-Fold Cross-Validation Results ({len(all_labels)} total samples)")
    print(f"{'='*60}")

    for fr in fold_results:
        print(f"  Fold {fr['fold']}: Acc={fr['accuracy']:.4f}  F1={fr['macro_f1']:.4f}  (n={fr['n']})")

    accs = [fr["accuracy"] for fr in fold_results]
    f1s = [fr["macro_f1"] for fr in fold_results]
    print(f"\n  Mean Accuracy: {np.mean(accs):.4f} +/- {np.std(accs):.4f}")
    print(f"  Mean Macro F1: {np.mean(f1s):.4f} +/- {np.std(f1s):.4f}")

    # Overall (all predictions pooled)
    overall_acc = accuracy_score(all_labels, all_preds)
    overall_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    precision, recall, f1_per, _ = precision_recall_fscore_support(
        all_labels, all_preds, labels=list(range(len(CLASS_NAMES))), zero_division=0,
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(CLASS_NAMES))))

    print(f"\n  Pooled Accuracy: {overall_acc:.4f}")
    print(f"  Pooled Macro F1: {overall_f1:.4f}")
    print(f"\n  Per-class (pooled across all folds):")
    for i, name in enumerate(CLASS_NAMES):
        print(f"    {name:8s}  P={precision[i]:.4f}  R={recall[i]:.4f}  F1={f1_per[i]:.4f}")

    print(f"\n  Confusion Matrix (pooled):")
    print(cm)

    # Restore original 80/20 split
    from material_classifier.label import assign_splits
    assign_splits(labels_csv)
    print(f"\nRestored original 80/20 train/val split in {labels_csv}")


if __name__ == "__main__":
    main()
