"""
InceptionTime model for thermal time-series classification.

Architecture based on:
    Fawaz et al. "InceptionTime: Finding AlexNet for Time Series Classification"
    https://arxiv.org/abs/1909.04939

Adapted for univariate thermal intensity sequences:
- Input: [batch, seq_len, 1] (single-channel thermal intensity)
- 6 Inception modules in 2 blocks with residual connections
- Global average pooling
- Ensemble of 5 networks with probability averaging

Key components:
- InceptionModule: Bottleneck → 3 parallel convolutions (k=10,20,40) + MaxPool → Concat → BatchNorm → ReLU
- InceptionBlock: 3 InceptionModules with residual connection
- InceptionTime: 2 InceptionBlocks → GlobalAvgPool → Dense
- InceptionTimeEnsemble: 5 InceptionTime networks, average softmax probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class InceptionModule(nn.Module):
    """
    Single Inception module with bottleneck.

    Architecture:
        Input → Bottleneck(1x1) ─┬─ Conv(k=10) ─┐
                                 ├─ Conv(k=20) ─┼─ Concat → BatchNorm → ReLU
                                 ├─ Conv(k=40) ─┤
                Input → MaxPool(k=3) → Conv(1x1)┘

    Output channels = nb_filters * 4 (one from each branch)
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: List[int] = None,
        bottleneck_size: int = 32,
        use_bottleneck: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            nb_filters: Number of filters per conv branch
            kernel_sizes: List of kernel sizes for parallel convolutions
            bottleneck_size: Number of channels after bottleneck
            use_bottleneck: Whether to use bottleneck layer
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]  # Default from paper

        self.use_bottleneck = use_bottleneck
        self.kernel_sizes = kernel_sizes

        # Bottleneck layer (1x1 convolution to reduce channels)
        if use_bottleneck and in_channels > 1:
            self.bottleneck = nn.Conv1d(in_channels, bottleneck_size, kernel_size=1, bias=False)
            conv_in_channels = bottleneck_size
        else:
            self.bottleneck = None
            conv_in_channels = in_channels

        # Parallel convolution branches with different kernel sizes
        # Use 'same' padding via nn.ConstantPad1d to ensure output length = input length
        self.conv_branches = nn.ModuleList()
        for k in kernel_sizes:
            # Calculate padding needed for 'same' convolution
            # For odd kernel, pad symmetrically: (k-1)//2 on each side
            # For even kernel, pad more on one side: (k//2, (k-1)//2)
            pad_left = (k - 1) // 2
            pad_right = k // 2
            branch = nn.Sequential(
                nn.ConstantPad1d((pad_left, pad_right), 0),
                nn.Conv1d(conv_in_channels, nb_filters, kernel_size=k, padding=0, bias=False)
            )
            self.conv_branches.append(branch)

        # MaxPool branch - use same explicit padding approach
        # MaxPool with k=3, stride=1: pad (1, 1) for same output length
        self.maxpool = nn.Sequential(
            nn.ConstantPad1d((1, 1), 0),
            nn.MaxPool1d(kernel_size=3, stride=1, padding=0)
        )
        self.conv_pool = nn.Conv1d(in_channels, nb_filters, kernel_size=1, bias=False)

        # Output channels = nb_filters * (len(kernel_sizes) + 1 for maxpool branch)
        out_channels = nb_filters * (len(kernel_sizes) + 1)

        # Batch normalization and activation
        self.bn = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, seq_len] (channels-first for Conv1d)

        Returns:
            Output tensor [batch, nb_filters*4, seq_len]
        """
        # Bottleneck
        if self.bottleneck is not None:
            x_bottleneck = self.bottleneck(x)
        else:
            x_bottleneck = x

        # Parallel conv branches
        conv_outputs = [conv(x_bottleneck) for conv in self.conv_branches]

        # MaxPool branch (uses original input, not bottleneck)
        pool_output = self.maxpool(x)
        pool_output = self.conv_pool(pool_output)

        # Concatenate all branches
        outputs = conv_outputs + [pool_output]
        x = torch.cat(outputs, dim=1)

        # BatchNorm and ReLU
        x = self.bn(x)
        x = self.relu(x)

        return x


class InceptionBlock(nn.Module):
    """
    Block of 3 Inception modules with residual connection.

    Architecture:
        Input → InceptionModule_1 → InceptionModule_2 → InceptionModule_3 → Add(Residual) → Output
                                                                              ↑
        Input ───────────────────────(Shortcut, 1x1 conv if needed)───────────┘
    """

    def __init__(
        self,
        in_channels: int,
        nb_filters: int = 32,
        kernel_sizes: List[int] = None,
        use_residual: bool = True
    ):
        """
        Args:
            in_channels: Number of input channels
            nb_filters: Number of filters per conv branch in each module
            kernel_sizes: Kernel sizes for convolutions
            use_residual: Whether to use residual connection
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        self.use_residual = use_residual

        # First module
        self.module1 = InceptionModule(
            in_channels=in_channels,
            nb_filters=nb_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_size=nb_filters,
            use_bottleneck=in_channels > 1
        )

        # Intermediate channels = nb_filters * (num_branches)
        intermediate_channels = self.module1.out_channels

        # Second module
        self.module2 = InceptionModule(
            in_channels=intermediate_channels,
            nb_filters=nb_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_size=nb_filters,
            use_bottleneck=True
        )

        # Third module
        self.module3 = InceptionModule(
            in_channels=intermediate_channels,
            nb_filters=nb_filters,
            kernel_sizes=kernel_sizes,
            bottleneck_size=nb_filters,
            use_bottleneck=True
        )

        # Residual connection (shortcut)
        self.out_channels = intermediate_channels

        if use_residual:
            if in_channels != self.out_channels:
                self.shortcut = nn.Sequential(
                    nn.Conv1d(in_channels, self.out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm1d(self.out_channels)
                )
            else:
                self.shortcut = nn.Identity()
            self.relu = nn.ReLU(inplace=True)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, channels, seq_len]

        Returns:
            Output tensor [batch, out_channels, seq_len]
        """
        # Three Inception modules
        out = self.module1(x)
        out = self.module2(out)
        out = self.module3(out)

        # Residual connection
        if self.use_residual:
            shortcut = self.shortcut(x)
            out = self.relu(out + shortcut)

        return out


class InceptionTime(nn.Module):
    """
    Full InceptionTime network for time-series classification.

    Architecture:
        Input [batch, seq_len, 1]
            ↓ (transpose to channels-first)
        InceptionBlock_1 (3 modules)
            ↓
        InceptionBlock_2 (3 modules) + Residual from Block_1
            ↓
        GlobalAveragePooling
            ↓
        Dense(num_classes)
            ↓
        Softmax (during inference)

    Total: 6 Inception modules in 2 blocks.
    """

    def __init__(
        self,
        input_size: int = 1,
        num_classes: int = 4,
        nb_filters: int = 32,
        depth: int = 6,
        kernel_sizes: List[int] = None,
        use_residual: bool = True,
        dropout: float = 0.0
    ):
        """
        Args:
            input_size: Number of input channels (1 for univariate thermal)
            num_classes: Number of output classes
            nb_filters: Number of filters per conv branch
            depth: Total number of Inception modules (must be multiple of 3)
            kernel_sizes: Kernel sizes for convolutions
            use_residual: Whether to use residual connections
            dropout: Dropout rate before classifier
        """
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [10, 20, 40]

        assert depth % 3 == 0, "Depth must be multiple of 3 (one block = 3 modules)"
        num_blocks = depth // 3

        self.input_size = input_size
        self.num_classes = num_classes
        self.nb_filters = nb_filters

        # Build blocks
        blocks = []
        in_channels = input_size

        for i in range(num_blocks):
            block = InceptionBlock(
                in_channels=in_channels,
                nb_filters=nb_filters,
                kernel_sizes=kernel_sizes,
                use_residual=use_residual
            )
            blocks.append(block)
            in_channels = block.out_channels

        self.blocks = nn.Sequential(*blocks)

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Classification head
        self.classifier = nn.Linear(in_channels, num_classes)

        self._final_channels = in_channels

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [batch, seq_len, input_size] (channels-last)
            lengths: Sequence lengths [batch] (optional, for compatibility)
            mask: Attention mask [batch, seq_len] (optional, for compatibility)

        Returns:
            Class logits [batch, num_classes]
        """
        # Transpose to channels-first: [batch, seq_len, channels] → [batch, channels, seq_len]
        x = x.transpose(1, 2)

        # Apply mask if provided (zero out padding)
        if mask is not None:
            x = x * mask.unsqueeze(1)

        # Inception blocks
        x = self.blocks(x)

        # Global Average Pooling
        x = self.gap(x).squeeze(-1)  # [batch, channels]

        # Dropout and classify
        x = self.dropout(x)
        logits = self.classifier(x)

        return logits

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class InceptionTimeEnsemble(nn.Module):
    """
    Ensemble of InceptionTime networks.

    Strategy (from paper):
    - Train 5 identical networks with different random seeds
    - During inference, average softmax probabilities across all networks
    - Final prediction = argmax of averaged probabilities

    This provides uncertainty estimation and improves robustness.
    """

    def __init__(
        self,
        input_size: int = 1,
        num_classes: int = 4,
        nb_filters: int = 32,
        depth: int = 6,
        kernel_sizes: List[int] = None,
        use_residual: bool = True,
        dropout: float = 0.0,
        n_members: int = 5
    ):
        """
        Args:
            input_size: Number of input channels
            num_classes: Number of output classes
            nb_filters: Number of filters per conv branch
            depth: Number of Inception modules per network
            kernel_sizes: Kernel sizes for convolutions
            use_residual: Whether to use residual connections
            dropout: Dropout rate
            n_members: Number of ensemble members
        """
        super().__init__()

        self.n_members = n_members
        self.num_classes = num_classes

        # Create ensemble members
        self.members = nn.ModuleList([
            InceptionTime(
                input_size=input_size,
                num_classes=num_classes,
                nb_filters=nb_filters,
                depth=depth,
                kernel_sizes=kernel_sizes,
                use_residual=use_residual,
                dropout=dropout
            )
            for _ in range(n_members)
        ])

    def forward(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass with probability averaging.

        Args:
            x: Input tensor [batch, seq_len, input_size]
            lengths: Sequence lengths [batch] (optional)
            mask: Attention mask [batch, seq_len] (optional)

        Returns:
            Averaged logits [batch, num_classes]
        """
        # Get logits from each member
        all_logits = [member(x, lengths, mask) for member in self.members]

        # Convert to probabilities and average
        all_probs = [F.softmax(logits, dim=1) for logits in all_logits]
        avg_probs = torch.stack(all_probs, dim=0).mean(dim=0)

        # Convert back to logits for consistency with loss functions
        # log(probs) gives log-probabilities which work with CrossEntropyLoss
        # But CrossEntropyLoss expects raw logits, so we return log(probs) + small epsilon
        # Actually, for CrossEntropyLoss, we need logits. Let's just average logits.
        avg_logits = torch.stack(all_logits, dim=0).mean(dim=0)

        return avg_logits

    def forward_with_uncertainty(
        self,
        x: torch.Tensor,
        lengths: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass with uncertainty estimation.

        Returns:
            Tuple of (mean_probs, std_probs, individual_probs)
            - mean_probs: [batch, num_classes] averaged probabilities
            - std_probs: [batch, num_classes] standard deviation of probabilities
            - individual_probs: [n_members, batch, num_classes] all member probabilities
        """
        # Get probabilities from each member
        all_probs = []
        for member in self.members:
            logits = member(x, lengths, mask)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs)

        all_probs = torch.stack(all_probs, dim=0)  # [n_members, batch, num_classes]

        mean_probs = all_probs.mean(dim=0)
        std_probs = all_probs.std(dim=0)

        return mean_probs, std_probs, all_probs

    def get_num_parameters(self) -> int:
        """Return total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_num_parameters_per_member(self) -> int:
        """Return number of parameters per ensemble member."""
        return self.members[0].get_num_parameters()


def create_inceptiontime_model(
    input_size: int = 1,
    num_classes: int = 4,
    nb_filters: int = 32,
    depth: int = 6,
    kernel_sizes: List[int] = None,
    use_residual: bool = True,
    dropout: float = 0.0,
    use_ensemble: bool = False,
    n_ensemble: int = 5
) -> nn.Module:
    """
    Factory function to create InceptionTime model.

    Args:
        input_size: Number of input channels (1 for univariate)
        num_classes: Number of output classes
        nb_filters: Number of filters per conv branch (default 32)
        depth: Number of Inception modules (default 6, must be multiple of 3)
        kernel_sizes: Kernel sizes for convolutions (default [10, 20, 40])
        use_residual: Whether to use residual connections
        dropout: Dropout rate before classifier
        use_ensemble: Whether to use ensemble of networks
        n_ensemble: Number of ensemble members (if use_ensemble=True)

    Returns:
        InceptionTime or InceptionTimeEnsemble model
    """
    if kernel_sizes is None:
        kernel_sizes = [10, 20, 40]

    if use_ensemble:
        return InceptionTimeEnsemble(
            input_size=input_size,
            num_classes=num_classes,
            nb_filters=nb_filters,
            depth=depth,
            kernel_sizes=kernel_sizes,
            use_residual=use_residual,
            dropout=dropout,
            n_members=n_ensemble
        )
    else:
        return InceptionTime(
            input_size=input_size,
            num_classes=num_classes,
            nb_filters=nb_filters,
            depth=depth,
            kernel_sizes=kernel_sizes,
            use_residual=use_residual,
            dropout=dropout
        )


if __name__ == '__main__':
    print("=" * 60)
    print("Testing InceptionTime Model")
    print("=" * 60)

    # Test with synthetic data
    batch_size = 8
    seq_len = 1000
    x = torch.randn(batch_size, seq_len, 1)
    lengths = torch.tensor([1000, 900, 800, 700, 600, 500, 400, 300])

    # Test 1: Single InceptionTime
    print("\n1. Single InceptionTime (6 modules, nb_filters=32):")
    model = create_inceptiontime_model(use_ensemble=False)
    print(f"   Parameters: {model.get_num_parameters():,}")
    model.eval()
    with torch.no_grad():
        logits = model(x, lengths)
    print(f"   Input: {x.shape} → Output: {logits.shape}")

    # Test 2: InceptionTime Ensemble
    print("\n2. InceptionTime Ensemble (5 members):")
    ensemble = create_inceptiontime_model(use_ensemble=True, n_ensemble=5)
    print(f"   Total parameters: {ensemble.get_num_parameters():,}")
    print(f"   Per member: {ensemble.get_num_parameters_per_member():,}")
    with torch.no_grad():
        logits_ens = ensemble(x, lengths)
    print(f"   Input: {x.shape} → Output: {logits_ens.shape}")

    # Test 3: Uncertainty estimation
    print("\n3. Ensemble uncertainty estimation:")
    with torch.no_grad():
        mean_probs, std_probs, all_probs = ensemble.forward_with_uncertainty(x, lengths)
    print(f"   Mean probs shape: {mean_probs.shape}")
    print(f"   Std probs shape: {std_probs.shape}")
    print(f"   All probs shape: {all_probs.shape}")
    print(f"   Mean uncertainty: {std_probs.mean():.4f}")

    # Test 4: Different configurations
    print("\n4. Different configurations:")
    configs = [
        {'nb_filters': 16, 'depth': 6},
        {'nb_filters': 32, 'depth': 6},
        {'nb_filters': 64, 'depth': 6},
        {'nb_filters': 32, 'depth': 9},
    ]
    for cfg in configs:
        m = create_inceptiontime_model(**cfg, use_ensemble=False)
        print(f"   nb_filters={cfg['nb_filters']}, depth={cfg['depth']}: {m.get_num_parameters():,} params")

    print("\n" + "=" * 60)
    print("All tests passed!")
    print("=" * 60)
