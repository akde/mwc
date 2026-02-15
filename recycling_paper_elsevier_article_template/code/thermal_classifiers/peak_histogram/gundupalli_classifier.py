#!/usr/bin/env python3
"""
Gundupalli et al. (2017) Peak Histogram Classifier

Implements the method from:
"Thermal imaging-based classification of the municipal solid waste
components" - Waste Management 2017

Method:
1. Extract mean (μ) and std (σ) from peak thermal frame
2. Compute class centroids from training data
3. Classify test samples by nearest centroid (Euclidean distance)

Usage:
    python gundupalli_classifier.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


def load_thermal_timeseries(csv_path: str):
    """
    Load thermal time-series data and extract peak statistics.

    For each track:
    - mean: peak thermal intensity (max value in time-series)
    - std: standard deviation of thermal values around peak
    """
    df = pd.read_csv(csv_path)

    features = []
    labels = []

    for _, row in df.iterrows():
        # Parse space-separated thermal values
        values = np.array([float(x) for x in row['mean_thermal_intensity_array'].split()])

        # Find peak frame and extract statistics
        peak_idx = np.argmax(values)
        peak_mean = values[peak_idx]

        # Compute std around peak (±50 frames window, or full series if shorter)
        window = 50
        start = max(0, peak_idx - window)
        end = min(len(values), peak_idx + window + 1)
        peak_std = np.std(values[start:end])

        features.append([peak_mean, peak_std])
        labels.append(row['class'])

    return np.array(features, dtype=np.float64), np.array(labels)


def train_centroids(X_train, y_train):
    """Compute class centroids (mean of each class)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    classes = np.unique(y_train)
    centroids = {}
    for c in classes:
        mask = y_train == c
        centroids[c] = X_scaled[mask].mean(axis=0)

    return centroids, scaler


def predict(X_test, centroids, scaler):
    """Classify by nearest centroid."""
    X_scaled = scaler.transform(X_test)
    predictions = []

    for x in X_scaled:
        min_dist = float('inf')
        best_class = None
        for c, centroid in centroids.items():
            dist = np.linalg.norm(x - centroid)
            if dist < min_dist:
                min_dist = dist
                best_class = c
        predictions.append(best_class)

    return np.array(predictions)


def main():
    CODE_DIR = Path(__file__).resolve().parent.parent.parent

    # Paths to train/validation/test datasets
    train_path = CODE_DIR / 'data' / 'final_thermal_train_dataset.csv'
    val_path = CODE_DIR / 'data' / 'final_thermal_validation_dataset.csv'
    test_path = CODE_DIR / 'data' / 'final_thermal_test_dataset.csv'

    print("=" * 60)
    print("GUNDUPALLI et al. (2017) CLASSIFIER")
    print("Features: peak mean, peak std (2D)")
    print("Method: Nearest Centroid")
    print("=" * 60)

    # Load training data
    print(f"\nLoading training data: {train_path}")
    X_train, y_train = load_thermal_timeseries(str(train_path))
    print(f"  Train samples: {len(X_train)}")
    print(f"  Classes: {dict(zip(*np.unique(y_train, return_counts=True)))}")

    # Load validation data
    print(f"\nLoading validation data: {val_path}")
    X_val, y_val = load_thermal_timeseries(str(val_path))
    print(f"  Validation samples: {len(X_val)}")
    print(f"  Classes: {dict(zip(*np.unique(y_val, return_counts=True)))}")

    # Load test data
    print(f"\nLoading test data: {test_path}")
    X_test, y_test = load_thermal_timeseries(str(test_path))
    print(f"  Test samples: {len(X_test)}")
    print(f"  Classes: {dict(zip(*np.unique(y_test, return_counts=True)))}")

    # Train (compute centroids)
    print("\nComputing class centroids...")
    centroids, scaler = train_centroids(X_train, y_train)

    print("\nClass centroids (standardized):")
    for c, centroid in centroids.items():
        print(f"  {c}: peak_mean={centroid[0]:.3f}, peak_std={centroid[1]:.3f}")

    # Predict on validation set
    y_val_pred = predict(X_val, centroids, scaler)

    # Predict on test set
    y_test_pred = predict(X_test, centroids, scaler)

    # Evaluate
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

    # Validation results
    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')

    print("\n" + "=" * 60)
    print("VALIDATION RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {val_accuracy:.4f}")
    print(f"F1 Weighted: {val_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_val_pred))

    print("Confusion Matrix:")
    val_classes = sorted(np.unique(np.concatenate([y_val, y_val_pred])))
    cm_val = confusion_matrix(y_val, y_val_pred, labels=val_classes)
    print(pd.DataFrame(cm_val, index=val_classes, columns=val_classes))

    # Test results
    test_accuracy = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')

    print("\n" + "=" * 60)
    print("TEST RESULTS")
    print("=" * 60)
    print(f"\nAccuracy: {test_accuracy:.4f}")
    print(f"F1 Weighted: {test_f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred))

    print("Confusion Matrix:")
    test_classes = sorted(np.unique(np.concatenate([y_test, y_test_pred])))
    cm_test = confusion_matrix(y_test, y_test_pred, labels=test_classes)
    print(pd.DataFrame(cm_test, index=test_classes, columns=test_classes))


if __name__ == '__main__':
    main()
