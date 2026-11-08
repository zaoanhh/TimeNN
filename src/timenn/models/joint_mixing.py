"""
JointFactorizedMixing: Joint Time-Frequency Scattering Mixing Layer

This module implements the Joint Time-Frequency Scattering theory from Andén et al. (2019).
It extends the original FactorizedMixing by applying 2D convolution on the time-frequency
representation, enabling the capture of Doppler frequency shift features (Chirp/Slope) in
CSI signals.

Key Design Principles:
1. Joint Convolution: Mixes both scale (frequency) and time dimensions simultaneously
2. Group Isolation: groups=c_in ensures different CSI subcarriers remain independent
3. Replicate Padding: Reduces boundary artifacts common in wavelet scattering
4. Scale Reduction: Optional pooling/projection to reduce parameters after 2D convolution

Reference:
    Andén, J., & Mallat, S. (2019). Deep Scattering Spectrum. IEEE Transactions on
    Signal Processing, 62(16), 4114-4128.

Author: LMS-Net Implementation
"""

from typing import Literal, Optional

import torch
import torch.nn as nn
from torch.nn.utils.parametrizations import spectral_norm


class JointFactorizedMixing(nn.Module):
    """
    Joint Time-Frequency Scattering Mixing Layer with Scale Reduction.

    This module implements the joint time-frequency scattering by treating the input
    as a 2D time-frequency representation (scalogram) and applying 2D convolution.

    Architecture:
        1. Reshape: [B, C_in * num_scales, T] -> [B, C_in, num_scales, T]
        2. Intra-subcarrier mixing (2D Conv): Mix scales and time jointly
           - groups = mixing_groups (configurable, defaults to c_in)
        3. GELU activation
        4. Scale Reduction (Optional): Reduce num_scales dimension to avoid parameter explosion
           - 'mean': Average pooling across scales
           - 'max': Max pooling across scales
           - 'learnable': Weighted sum via depthwise convolution
        5. Reshape: [B, C_new, num_scales, T] -> [B, C_new * num_scales, T] (if not reduced)
        6. Inter-subcarrier mixing (1x1 Conv): Standard factorized mixing

    Args:
        in_channels: Total input channels (C_in * num_scales).
        c_in: Original number of subcarrier/sensor channels.
        num_scales: Number of frequency scales per subcarrier.
        intra_out_per_group: Output channels per group for intra-subcarrier mixing.
        inter_out: Output channels for inter-subcarrier mixing.
        freq_kernel_size: Convolution kernel size in frequency dimension (scales).
            Default: 3 (captures local scale relationships).
        time_kernel_size: Convolution kernel size in time dimension.
            Default: 15 (captures long-range acceleration/Doppler features).
        activation: Activation function between steps ('gelu', 'relu', 'none').
        reduce_scales: Whether to reduce scale dimension before inter_conv.
            When True, significantly reduces parameters. Default: True.
        pool_type: Method for scale reduction if reduce_scales=True.
            Options: 'mean', 'max', 'learnable'. Default: 'learnable'.
        mixing_groups: Number of groups for intra-convolution.
            - If None (default), uses c_in for subcarrier isolation.
            - If set to a specific value, uses that value for groups.
            - For Stage 1 (c_in=physical subcarriers): use c_in or None.
            - For Stage 2+ (c_in=intermediate channels): use smaller value like 8, 16, 32.

    Input shape: [Batch, in_channels, Length] where in_channels = C_in * num_scales
    Output shape: [Batch, inter_out, Length]

    Example:
        >>> # Input: B=2, C_in=3, num_scales=4, T=100
        >>> x = torch.randn(2, 3 * 4, 100)  # [B, 12, 100]
        >>> mixing = JointFactorizedMixing(
        ...     in_channels=12, c_in=3, num_scales=4,
        ...     intra_out_per_group=8, inter_out=64,
        ...     reduce_scales=True, pool_type='learnable'
        ... )
        >>> out = mixing(x)  # [B, 64, 100]
    """

    def __init__(
        self,
        in_channels: int,
        c_in: int,
        num_scales: int,
        intra_out_per_group: int,
        inter_out: int,
        freq_kernel_size: int = 3,
        time_kernel_size: int = 15,
        activation: Literal["gelu", "relu", "none"] = "gelu",
        reduce_scales: bool = True,
        pool_type: Literal["mean", "max", "learnable"] = "learnable",
        mixing_groups: Optional[int] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.num_scales = num_scales
        self.in_channels = in_channels
        self.reduce_scales = reduce_scales
        self.pool_type = pool_type
        # Use mixing_groups if provided, otherwise use c_in
        self.mixing_groups = mixing_groups if mixing_groups is not None else c_in

        # Validate channel configuration
        assert in_channels % c_in == 0, (
            f"in_channels ({in_channels}) must be divisible by c_in ({c_in})"
        )
        assert in_channels % num_scales == 0, (
            f"in_channels ({in_channels}) must be divisible by num_scales ({num_scales})"
        )
        # Validate mixing_groups
        assert in_channels % self.mixing_groups == 0, (
            f"in_channels ({in_channels}) must be divisible by mixing_groups ({self.mixing_groups})"
        )

        # Calculate output channels for intra-subcarrier mixing
        # intra_total_out = c_in * intra_out_per_group
        intra_total_out = self.mixing_groups * intra_out_per_group
        self.intra_total_out = intra_total_out

        # ==========================================================================
        # [Check 1]: Padding Calculation for Same Output Shape
        # ==========================================================================
        # For separable convolutions, calculate padding for each dimension:
        #   padding_h = (K_h - 1) // 2  (for height/frequency dimension)
        #   padding_w = (K_w - 1) // 2  (for width/time dimension)
        #
        # Layer 1: Only pad frequency dimension (height)
        # Layer 2: Only pad time dimension (width)
        # ==========================================================================

        pad_freq = (freq_kernel_size - 1) // 2
        pad_time = (time_kernel_size - 1) // 2

        # ==========================================================================
        # Separable Joint Scattering (Andén et al., 2019)
        # ==========================================================================
        # Factorize the joint 2D convolution into two separable 1D convolutions:
        #   Ψ(t, λ) = ψ(t) · ψ(λ)  (wavelet product)
        #
        # This significantly reduces parameters:
        #   Full 2D: c_in * intra_total_out * freq_kernel_size * time_kernel_size
        #   Separable: c_in * intra_total_out * freq_kernel_size + intra_total_out * time_kernel_size
        #
        # Layer 1 (Frequency Mixing): Conv2d with kernel_size=(freq_kernel_size, 1)
        #   - Handles channel expansion from c_in to intra_total_out
        #   - groups=c_in ensures subcarrier independence
        #   - padding=(pad_freq, 0) preserves frequency dimension
        #
        # Layer 2 (Time Mixing): Conv2d with kernel_size=(1, time_kernel_size)
        #   - Depthwise convolution (groups=intra_total_out) minimizes parameters
        #   - padding=(0, pad_time) preserves time dimension
        # ==========================================================================

        # Step 1: Intra-subcarrier mixing with Separable 2D Convolution
        # Input: [B, c_in, num_scales, T] -> Output: [B, intra_total_out, num_scales, T]
        self.intra_conv = nn.Sequential(
            # Frequency mixing: expands channels while mixing scales
            spectral_norm(
                nn.Conv2d(
                    in_channels=c_in,
                    out_channels=intra_total_out,
                    kernel_size=(freq_kernel_size, 1),
                    stride=(1, 1),
                    padding=(pad_freq, 0),
                    padding_mode="replicate",
                    bias=False,
                    groups=c_in,
                )
            ),
            # Time mixing: depthwise convolution along time dimension
            spectral_norm(
                nn.Conv2d(
                    in_channels=intra_total_out,
                    out_channels=intra_total_out,
                    kernel_size=(1, time_kernel_size),
                    stride=(1, 1),
                    padding=(0, pad_time),
                    padding_mode="replicate",
                    bias=True,
                    groups=intra_total_out,
                )
            ),
        )

        # Activation between steps
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        # ==========================================================================
        # Scale Reduction Layer (Optional)
        # ==========================================================================
        # According to Andén et al. (2019), after extracting time-frequency features,
        # Pooling across scales provides frequency-shift invariance and
        # significantly reduce parameters.
        #
        # Without reduction: inter_in_channels = intra_total_out * num_scales
        # With reduction:    inter_in_channels = intra_total_out
        # ==========================================================================

        self.scale_proj = None
        if reduce_scales:
            if pool_type == "learnable":
                # Learnable weighted sum via depthwise convolution
                # kernel_size=(num_scales, 1) sums across scales with learned weights
                # groups=intra_total_out ensures each channel has its own weights
                self.scale_proj = spectral_norm(
                    nn.Conv2d(
                        in_channels=intra_total_out,
                        out_channels=intra_total_out,
                        kernel_size=(num_scales, 1),
                        stride=(1, 1),
                        padding=(0, 0),  # No padding needed for full reduction
                        groups=intra_total_out,  # Depthwise: each channel has own weights
                        bias=True,
                    )
                )
            # For 'mean' and 'max', no learnable parameters needed

        # Input channels of inter_conv depend on whether scales are reduced.
        if reduce_scales:
            inter_in_channels = intra_total_out  # Scales are aggregated
        else:
            inter_in_channels = intra_total_out * num_scales  # Keep all scales in channels

        self.inter_in_channels = inter_in_channels

        # ==========================================================================
        # Step 4: Inter-subcarrier Mixing (1x1 Conv1d)
        # ==========================================================================
        # Mix information across different subcarriers
        self.inter_conv = spectral_norm(
            nn.Conv1d(
                in_channels=inter_in_channels,
                out_channels=inter_out,
                kernel_size=1,
                bias=True,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply joint time-frequency mixing.

        Args:
            x: Input tensor of shape [Batch, in_channels, Length]
               where in_channels = C_in * num_scales

        Returns:
            Output tensor of shape [Batch, inter_out, Length]
        """
        B, C, L = x.shape

        # ==========================================================================
        # Step 1: Reshape to Time-Frequency Representation
        # ==========================================================================
        # [B, C_in * num_scales, L] -> [B, C_in, num_scales, L]
        # This treats each subcarrier's scales as a frequency axis
        x_reshaped = x.view(B, self.c_in, self.num_scales, L)

        # ==========================================================================
        # Step 2: Intra-subcarrier Mixing (Joint 2D Convolution)
        # ==========================================================================
        # Apply 2D convolution on (scale, time) dimensions
        # Conv2d input: [B, C_in, num_scales, L]
        # Conv2d output: [B, C_in * intra_out_per_group, num_scales, L]
        out = self.intra_conv(x_reshaped)

        # Activation
        out = self.act(out)

        # ==========================================================================
        # Step 3: Scale Reduction (Optional)
        # ==========================================================================
        # After the 2D convolution, the tensor is [B, intra_total_out, num_scales, L].
        # Reduce the scale dimension to limit the inter_conv parameter count.
        #
        # Without reduction: [B, intra_total_out * num_scales, L] -> inter_conv: 4096 -> 128
        # With reduction:    [B, intra_total_out, L]           -> inter_conv: 512 -> 128
        # ==========================================================================
        if self.reduce_scales:
            if self.pool_type == "mean":
                # Mean pooling across scales
                # [B, intra_total_out, num_scales, L] -> [B, intra_total_out, L]
                out = out.mean(dim=2)
            elif self.pool_type == "max":
                # Max pooling across scales
                # [B, intra_total_out, num_scales, L] -> [B, intra_total_out, L]
                out = out.max(dim=2)[0]
            elif self.pool_type == "learnable" and self.scale_proj is not None:
                # Learnable weighted sum via depthwise convolution
                # [B, intra_total_out, num_scales, L] -> [B, intra_total_out, 1, L]
                out = self.scale_proj(out)
                # Squeeze the scale dimension: [B, intra_total_out, 1, L] -> [B, intra_total_out, L]
                out = out.squeeze(2)
        else:
            # Flatten scales into channels
            # [B, intra_total_out * num_scales, L]
            out = out.flatten(start_dim=1, end_dim=2)

        # ==========================================================================
        # Step 4: Inter-subcarrier Mixing (1x1 Conv1d)
        # ==========================================================================
        # Mix information across different subcarriers
        out = self.inter_conv(out)

        return out


# ==============================================================================
# Convenience Factory Function
# ==============================================================================


def create_joint_factorized_mixing(
    in_channels: int,
    c_in: int,
    num_scales: int,
    intra_out_per_group: int,
    inter_out: int,
    freq_kernel_size: int = 3,
    time_kernel_size: int = 15,
    activation: Literal["gelu", "relu", "none"] = "gelu",
    reduce_scales: bool = True,
    pool_type: Literal["mean", "max", "learnable"] = "learnable",
    mixing_groups: Optional[int] = None,
) -> JointFactorizedMixing:
    """
    Factory function to create JointFactorizedMixing layer.

    This is a convenience wrapper that validates inputs and creates the module.

    Args:
        in_channels: Total input channels (C_in * num_scales).
        c_in: Original number of subcarrier channels.
        num_scales: Number of frequency scales per subcarrier.
        intra_out_per_group: Output channels per group for intra-subcarrier mixing.
        inter_out: Output channels for inter-subcarrier mixing.
        freq_kernel_size: Convolution kernel size in frequency dimension. Default: 3.
        time_kernel_size: Convolution kernel size in time dimension. Default: 15.
        activation: Activation function. Default: 'gelu'.
        reduce_scales: Whether to reduce scale dimension. Default: True.
        pool_type: Method for scale reduction. Default: 'learnable'.
        mixing_groups: Number of groups for intra-convolution. Default: None (uses c_in).

    Returns:
        Configured JointFactorizedMixing instance.
    """
    return JointFactorizedMixing(
        in_channels=in_channels,
        c_in=c_in,
        num_scales=num_scales,
        intra_out_per_group=intra_out_per_group,
        inter_out=inter_out,
        freq_kernel_size=freq_kernel_size,
        time_kernel_size=time_kernel_size,
        activation=activation,
        reduce_scales=reduce_scales,
        pool_type=pool_type,
        mixing_groups=mixing_groups,
    )
