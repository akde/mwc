"""
MiniRocket Multivariate Implementation

Adapted from the official implementation by Dempster et al.
https://github.com/angus924/minirocket

This is a numba-accelerated implementation for multivariate time series.
"""

import numpy as np
from numba import njit, prange

# =============================================================================
# CONSTANTS
# =============================================================================

# 84 fixed kernels (length 9, 3 values from {-1, 2})
# These are the only kernels that produce unique PPV features
INDICES = np.array([
    [0,1,2],[0,1,3],[0,1,4],[0,1,5],[0,1,6],[0,1,7],[0,1,8],
    [0,2,3],[0,2,4],[0,2,5],[0,2,6],[0,2,7],[0,2,8],
    [0,3,4],[0,3,5],[0,3,6],[0,3,7],[0,3,8],
    [0,4,5],[0,4,6],[0,4,7],[0,4,8],
    [0,5,6],[0,5,7],[0,5,8],
    [0,6,7],[0,6,8],
    [0,7,8],
    [1,2,3],[1,2,4],[1,2,5],[1,2,6],[1,2,7],[1,2,8],
    [1,3,4],[1,3,5],[1,3,6],[1,3,7],[1,3,8],
    [1,4,5],[1,4,6],[1,4,7],[1,4,8],
    [1,5,6],[1,5,7],[1,5,8],
    [1,6,7],[1,6,8],
    [1,7,8],
    [2,3,4],[2,3,5],[2,3,6],[2,3,7],[2,3,8],
    [2,4,5],[2,4,6],[2,4,7],[2,4,8],
    [2,5,6],[2,5,7],[2,5,8],
    [2,6,7],[2,6,8],
    [2,7,8],
    [3,4,5],[3,4,6],[3,4,7],[3,4,8],
    [3,5,6],[3,5,7],[3,5,8],
    [3,6,7],[3,6,8],
    [3,7,8],
    [4,5,6],[4,5,7],[4,5,8],
    [4,6,7],[4,6,8],
    [4,7,8],
    [5,6,7],[5,6,8],
    [5,7,8],
    [6,7,8]
], dtype=np.int32)


# =============================================================================
# FITTING (Computing Quantiles)
# =============================================================================

@njit(fastmath=True, cache=True)
def _fit_biases(X, num_kernels=10000):
    """
    Compute bias values (quantiles of convolution outputs) for MiniRocket.

    Args:
        X: Training data of shape (n_samples, n_channels, n_timepoints)
        num_kernels: Number of kernels to use (default 10000)

    Returns:
        biases: Array of bias values, shape (num_kernels,)
        dilations: Array of dilation values
        num_features_per_dilation: Features per dilation level
        combinations: Channel combinations to use
    """
    n_samples, n_channels, n_timepoints = X.shape

    # Dilations: powers of 2 up to floor(log2((n_timepoints - 1) / 8))
    max_exponent = max(0, int(np.floor(np.log2((n_timepoints - 1) / 8))))
    dilations = np.array([2 ** i for i in range(max_exponent + 1)], dtype=np.int32)
    num_dilations = len(dilations)

    # Number of features per kernel
    num_features_per_kernel = 84  # Fixed kernels

    # Features per dilation (distribute evenly)
    num_kernels_per_dilation = num_kernels // num_dilations
    num_features_per_dilation = np.zeros(num_dilations, dtype=np.int32)
    for i in range(num_dilations):
        num_features_per_dilation[i] = num_kernels_per_dilation

    # Channel combinations (for multivariate)
    if n_channels > 1:
        num_channel_combinations = min(n_channels, 9)  # Max 9 channel combinations
        combinations = np.zeros((num_channel_combinations, n_channels), dtype=np.float32)
        for c in range(num_channel_combinations):
            combinations[c] = np.random.randn(n_channels).astype(np.float32)
    else:
        num_channel_combinations = 1
        combinations = np.ones((1, 1), dtype=np.float32)

    # Compute biases from quantiles
    total_features = num_kernels
    biases = np.zeros(total_features, dtype=np.float32)

    feature_idx = 0
    for dilation_idx in range(num_dilations):
        dilation = dilations[dilation_idx]
        padding = ((9 - 1) * dilation) // 2

        for _ in range(num_features_per_dilation[dilation_idx]):
            if feature_idx >= total_features:
                break

            # Sample random sample
            sample_idx = np.random.randint(0, n_samples)

            # Sample random channel combination
            comb_idx = np.random.randint(0, num_channel_combinations)

            # Sample random kernel (from 84 fixed patterns)
            kernel_idx = np.random.randint(0, 84)
            indices = INDICES[kernel_idx]

            # Combine channels
            combined = np.zeros(n_timepoints, dtype=np.float32)
            for c in range(n_channels):
                combined += combinations[comb_idx, c] * X[sample_idx, c, :]

            # Compute convolution at random position
            conv_values = []
            for t in range(padding, n_timepoints - padding - 8 * dilation + 1):
                val = -1.0
                for k, idx in enumerate(indices):
                    if k < 3:
                        val += 3 * combined[t + idx * dilation]
                conv_values.append(val)

            if len(conv_values) > 0:
                # Use random quantile as bias
                quantile = np.random.rand()
                sorted_vals = np.sort(np.array(conv_values))
                bias_idx = int(quantile * len(sorted_vals))
                bias_idx = min(bias_idx, len(sorted_vals) - 1)
                biases[feature_idx] = sorted_vals[bias_idx]

            feature_idx += 1

    return biases, dilations, num_features_per_dilation, combinations


def fit(X, num_kernels=10000, random_state=42):
    """
    Fit MiniRocket transform on training data.

    Args:
        X: Training data of shape (n_samples, n_channels, n_timepoints)
           OR (n_samples, n_timepoints) for univariate
        num_kernels: Number of kernels (default 10000)
        random_state: Random seed for reproducibility

    Returns:
        parameters: Dictionary containing fitted parameters
    """
    np.random.seed(random_state)

    # Handle 2D input (univariate)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    # Ensure float32
    X = X.astype(np.float32)

    n_samples, n_channels, n_timepoints = X.shape

    # Minimum length check
    if n_timepoints < 9:
        raise ValueError(f"Minimum sequence length is 9, got {n_timepoints}")

    biases, dilations, num_features_per_dilation, combinations = _fit_biases(
        X, num_kernels
    )

    return {
        'biases': biases,
        'dilations': dilations,
        'num_features_per_dilation': num_features_per_dilation,
        'combinations': combinations,
        'num_kernels': num_kernels,
        'n_channels': n_channels,
    }


# =============================================================================
# TRANSFORM (Computing PPV Features)
# =============================================================================

@njit(fastmath=True, parallel=True, cache=True)
def _transform(X, biases, dilations, num_features_per_dilation, combinations):
    """
    Apply MiniRocket transform to compute PPV features.

    Args:
        X: Data of shape (n_samples, n_channels, n_timepoints)
        biases: Fitted bias values
        dilations: Dilation values
        num_features_per_dilation: Features per dilation
        combinations: Channel combinations

    Returns:
        features: PPV features of shape (n_samples, num_features)
    """
    n_samples, n_channels, n_timepoints = X.shape
    num_features = len(biases)
    features = np.zeros((n_samples, num_features), dtype=np.float32)

    num_dilations = len(dilations)
    num_combinations = combinations.shape[0]

    for sample_idx in prange(n_samples):
        feature_idx = 0

        for dilation_idx in range(num_dilations):
            dilation = dilations[dilation_idx]
            padding = ((9 - 1) * dilation) // 2

            for feat_in_dilation in range(num_features_per_dilation[dilation_idx]):
                if feature_idx >= num_features:
                    break

                # Determine which combination and kernel to use
                comb_idx = feat_in_dilation % num_combinations
                kernel_idx = (feat_in_dilation // num_combinations) % 84
                indices = INDICES[kernel_idx]

                # Combine channels
                combined = np.zeros(n_timepoints, dtype=np.float32)
                for c in range(n_channels):
                    combined += combinations[comb_idx, c] * X[sample_idx, c, :]

                # Compute convolution and PPV
                count_positive = 0
                count_total = 0
                bias = biases[feature_idx]

                for t in range(padding, n_timepoints - padding - 8 * dilation + 1):
                    val = -1.0
                    for k in range(3):
                        idx = indices[k]
                        val += 3 * combined[t + idx * dilation]

                    count_total += 1
                    if val > bias:
                        count_positive += 1

                # PPV = proportion of positive values
                if count_total > 0:
                    features[sample_idx, feature_idx] = count_positive / count_total

                feature_idx += 1

    return features


def transform(X, parameters):
    """
    Transform data using fitted MiniRocket parameters.

    Args:
        X: Data of shape (n_samples, n_channels, n_timepoints)
           OR (n_samples, n_timepoints) for univariate
        parameters: Dictionary from fit()

    Returns:
        features: PPV features of shape (n_samples, num_kernels)
    """
    # Handle 2D input (univariate)
    if X.ndim == 2:
        X = X[:, np.newaxis, :]

    # Ensure float32
    X = X.astype(np.float32)

    n_samples, n_channels, n_timepoints = X.shape

    # Check channel count matches
    if n_channels != parameters['n_channels']:
        raise ValueError(
            f"Expected {parameters['n_channels']} channels, got {n_channels}"
        )

    # Minimum length check
    if n_timepoints < 9:
        raise ValueError(f"Minimum sequence length is 9, got {n_timepoints}")

    features = _transform(
        X,
        parameters['biases'],
        parameters['dilations'],
        parameters['num_features_per_dilation'],
        parameters['combinations'],
    )

    return features


# =============================================================================
# CONVENIENCE CLASS
# =============================================================================

class MiniRocketMultivariate:
    """
    MiniRocket for Multivariate Time Series Classification.

    Usage:
        from minirocket import MiniRocketMultivariate
        from sklearn.linear_model import RidgeClassifierCV

        # Transform
        rocket = MiniRocketMultivariate(num_kernels=10000)
        rocket.fit(X_train)
        X_train_transform = rocket.transform(X_train)
        X_test_transform = rocket.transform(X_test)

        # Classify
        classifier = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
        classifier.fit(X_train_transform, y_train)
        predictions = classifier.predict(X_test_transform)
    """

    def __init__(self, num_kernels=10000, random_state=42):
        self.num_kernels = num_kernels
        self.random_state = random_state
        self.parameters_ = None

    def fit(self, X):
        """Fit the transform on training data."""
        self.parameters_ = fit(X, self.num_kernels, self.random_state)
        return self

    def transform(self, X):
        """Transform data using fitted parameters."""
        if self.parameters_ is None:
            raise ValueError("Must call fit() before transform()")
        return transform(X, self.parameters_)

    def fit_transform(self, X):
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)


if __name__ == '__main__':
    # Quick test
    np.random.seed(42)

    # Create dummy multivariate data
    n_samples = 100
    n_channels = 4
    n_timepoints = 50

    X = np.random.randn(n_samples, n_channels, n_timepoints).astype(np.float32)

    # Test fit and transform
    rocket = MiniRocketMultivariate(num_kernels=1000, random_state=42)
    features = rocket.fit_transform(X)

    print(f"Input shape: {X.shape}")
    print(f"Output shape: {features.shape}")
    print(f"Feature range: [{features.min():.4f}, {features.max():.4f}]")
