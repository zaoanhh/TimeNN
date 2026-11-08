"""
LMS-Net: Learnable Mel-Scattering Network

A modular, highly configurable 1D time series classification network for CSI (Channel State Information)
and physical vibration signals.

Core Design Principles:
1. **Lipschitz Stability**: Physical scattering mechanism + smoothness constraints ensure stability
   against time deformations.
2. **Frequency Coverage**: Parseval regularization (Coverage Loss) forces filter bank to tile frequency
   domain without mode collapse (Frame Theory).
3. **Parameter Sharing**: Physical filters are shared across all C_in channels for consistent
   physical observations.

Theoretical Notes:
- Morse Wavelets provide asymmetric time-frequency localization through learnable γ (symmetry) and
  β (time-bandwidth) parameters, offering additional modeling capability for non-symmetric transient signals.
- Coverage Loss implements the Littlewood-Paley condition (Parseval Frame) to guarantee energy
  conservation and feature completeness.

Author: LMS-Net Implementation
"""

from dataclasses import dataclass
from typing import Literal, Optional, List, Tuple
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import spectral_norm

from timenn.models.joint_mixing import JointFactorizedMixing


# ==============================================================================
# 1. Configuration Module: FilterConfig
# ==============================================================================


@dataclass
class FilterConfig:
    """
    Configuration dataclass for controlling physical layer hyperparameters.

    Attributes:
        filter_type: Type of physical waveform:
            - 'morlet' (default): Gaussian-enveloped sinusoid, most robust.
            - 'chirplet': Linear frequency modulation wavelet for Doppler acceleration effects.
            - 'morse': Generalized Morse Wavelets with learnable asymmetric parameters
                       γ (symmetry) and β (time-bandwidth) for complex transient signals.
        scale_type: Frequency scale distribution:
            - 'mel': Biologically-inspired mel-scale spacing.
            - 'linear': Uniform spacing for mechanical/CSI signals.
        num_scales: Number of scales to extract per subcarrier (e.g., 4).
        min_sigma: Lower bound for bandwidth σ (e.g., 0.05) to ensure smoothness.
        f_min: Minimum center frequency (normalized, 0-0.5).
        f_max: Maximum center frequency (normalized, 0-0.5).
        kernel_length: Length of the filter kernel in samples.
        use_joint_mixing: Whether to use JointFactorizedMixing (2D time-frequency conv).
        freq_kernel_size: Convolution kernel size in frequency dimension for joint mixing.
        time_kernel_size: Convolution kernel size in time dimension for joint mixing.
        reduce_scales: Whether to reduce scale dimension before inter_conv (default: True).
        pool_type: Method for scale reduction ('mean', 'max', 'learnable', default: 'learnable').
        mixing_groups: Number of groups for mixing layer convolution.
            - If None (default), uses c_in for subcarrier isolation.
            - If set to a specific value, uses that value for groups.
            - For Stage 1 (c_in=subcarriers): use c_in or None.
            - For Stage 2+ (c_in=inter_out): use smaller value like 8, 16, or 32.
    """

    filter_type: Literal["morlet", "chirplet", "morse"] = "morlet"
    scale_type: Literal["mel", "linear"] = "mel"
    num_scales: int = 4
    min_sigma: float = 0.05
    f_min: float = 0.01
    f_max: float = 0.4
    kernel_length: int = 64
    use_joint_mixing: bool = False
    freq_kernel_size: int = 3
    time_kernel_size: int = 15
    reduce_scales: bool = True
    pool_type: Literal["mean", "max", "learnable"] = "learnable"
    mixing_groups: Optional[int] = None  # None means use c_in


# ==============================================================================
# 2. Core Module: PhysicalFilterBank
# ==============================================================================


class PhysicalFilterBank(nn.Module):
    """
    Parameterized, learnable complex filter layer.

    This module generates complex convolution kernels dynamically based on the filter
    configuration. The same set of filters is applied to all C_in channels (broadcasting trick)
    to ensure consistent physical observation across subcarriers.

    Key Features:
    - Learnable center frequencies (ξ) and bandwidths (σ)
    - Support for Morlet, Chirplet, and Morse wavelets
    - Smoothness constraint on σ to ensure Lipschitz stability
    - Coverage loss for Parseval frame regularization

    Input shape: [Batch, C_in, Length]
    Output shape: [Batch, C_in * num_scales, Length]
    """

    def __init__(
        self,
        config: FilterConfig,
        c_in: int,
    ):
        """
        Args:
            config: FilterConfig instance with filter hyperparameters.
            c_in: Number of input subcarrier/sensor channels.
        """
        super().__init__()
        self.config = config
        self.c_in = c_in
        self.num_scales = config.num_scales
        self.kernel_length = config.kernel_length
        self.min_sigma = config.min_sigma

        # Initialize delta_xi_raw via inverse operations to match target frequencies
        xi_target = self._init_center_frequencies()  # [num_scales]
        delta_xi_raw_init = self._inverse_xi_transform(xi_target)
        self.delta_xi_raw = nn.Parameter(delta_xi_raw_init)  # [num_scales]

        # Initialize sigma with Constant-Q property (kymatio style: sigma �propto xi)
        xi_detached = self._init_center_frequencies().detach()
        J_freq = np.log2(self.config.f_max / self.config.f_min)
        Q = self.num_scales / J_freq
        r = np.sqrt(0.5)
        factor = 1.0 / np.power(2.0, 1.0 / Q)
        term1 = (1.0 - factor) / (1.0 + factor)
        term2 = 1.0 / np.sqrt(2.0 * np.log(1.0 / r))
        sigma_init = xi_detached * term1 * term2
        sigma_clamped = torch.clamp(sigma_init, min=self.min_sigma)
        sigma_raw_init = torch.log1p(torch.exp(sigma_clamped - self.min_sigma) - 1.0)
        self.sigma_raw = nn.Parameter(sigma_raw_init)  # [num_scales]

        # Filter-specific parameters
        if config.filter_type == "morse":
            self.gamma = nn.Parameter(torch.ones(config.num_scales) * 3.0)
            self.beta = nn.Parameter(torch.ones(config.num_scales) * 5.0)
        elif config.filter_type == "chirplet":
            self.alpha = nn.Parameter(torch.zeros(config.num_scales))

    def _init_center_frequencies(self) -> torch.Tensor:
        """Initialize center frequencies based on scale_type."""
        num_scales = self.config.num_scales
        f_min = self.config.f_min
        f_max = self.config.f_max

        if self.config.scale_type == "mel":
            mel_min = 2595 * math.log10(1 + f_min * 16000 / 700)
            mel_max = 2595 * math.log10(1 + f_max * 16000 / 700)
            mels = torch.linspace(mel_min, mel_max, num_scales)
            freqs = 700 * (10 ** (mels / 2595) - 1) / 16000
        else:
            freqs = torch.linspace(f_min, f_max, num_scales)

        return freqs

    def _inverse_xi_transform(self, xi_target: torch.Tensor) -> torch.Tensor:
        """
        Compute delta_xi_raw values that produce xi_target after forward transform.

        Forward: delta_xi_raw -> softplus -> cumsum -> normalize to [0,1] -> scale to [f_min, f_max]
        """
        f_min = self.config.f_min
        f_max = self.config.f_max
        eps = 1e-6

        # Step 1: Inverse of linear scaling [f_min, f_max] -> [0, 1]
        normalized = (xi_target - f_min) / (f_max - f_min + eps)
        normalized = torch.clamp(normalized, min=eps, max=1.0 - eps)

        # Step 2: Inverse of normalization (multiply by final cumsum value)
        # Find cumsum values that normalize to these targets.
        # normalized = cumsum / cumsum[-1] => cumsum = normalized * cumsum[-1]
        # Choose cumsum[-1] = num_scales for reasonable scale
        target_cumsum_max = float(self.num_scales)
        cumsum_target = normalized * target_cumsum_max

        # Step 3: Inverse cumsum -> deltas
        deltas = torch.zeros_like(cumsum_target)
        deltas[0] = cumsum_target[0]
        if len(cumsum_target) > 1:
            deltas[1:] = cumsum_target[1:] - cumsum_target[:-1]
        deltas = torch.clamp(deltas, min=eps)

        # Step 4: Inverse softplus
        delta_xi_raw = torch.log(torch.exp(deltas) - 1.0 + eps)

        return delta_xi_raw

    @property
    def sigma(self) -> torch.Tensor:
        """Constrained bandwidth parameter (σ >= min_sigma for smoothness)."""
        return F.softplus(self.sigma_raw) + self.min_sigma

    @property
    def xi(self) -> torch.Tensor:
        """
        Compute strictly ordered and bounded center frequencies.

        Guarantees: f_min < xi_0 < xi_1 < ... < xi_{N-1} < f_max
        """
        # Step 1: Positive increments
        deltas = F.softplus(self.delta_xi_raw) + 1e-5

        # Step 2: Cumulative sum -> strictly increasing
        cumsum = torch.cumsum(deltas, dim=0)

        # Step 3: Normalize to [0, 1] then scale to [f_min, f_max]
        # Add small margin to avoid exact boundary values
        margin = 1e-4
        normalized = cumsum / (cumsum[-1] + 1e-8)
        xi = (
            self.config.f_min
            + margin
            + (self.config.f_max - self.config.f_min - 2 * margin) * normalized
        )

        return xi

    def _generate_morlet_kernel(self) -> torch.Tensor:
        """
        Generate Morlet wavelet kernels.

        Morlet: ψ(t) = (π * σ²)^(-1/4) * exp(-t²/(2σ²)) * exp(2πi * ξ * t)

        Returns:
            Complex kernels of shape [num_scales, kernel_length]
        """
        t = torch.linspace(
            -self.kernel_length // 2,
            self.kernel_length // 2,
            self.kernel_length,
            device=self.delta_xi_raw.device,
        )  # [kernel_length]

        sigma = self.sigma.unsqueeze(1)  # [num_scales, 1]
        xi = self.xi.unsqueeze(1)  # [num_scales, 1]
        t = t.unsqueeze(0)  # [1, kernel_length]

        # Gaussian envelope
        gaussian = torch.exp(-(t**2) / (2 * sigma**2))

        # Complex sinusoid
        phase = 2 * math.pi * xi * t
        sinusoid = torch.complex(torch.cos(phase), torch.sin(phase))

        # Normalize
        norm = (math.pi * sigma**2) ** (-0.25)

        kernel = norm * gaussian * sinusoid  # [num_scales, kernel_length]
        return kernel

    def _generate_chirplet_kernel(self) -> torch.Tensor:
        """
        Generate Chirplet wavelet kernels.

        Chirplet: ψ(t) = exp(-t²/(2σ²)) * exp(2πi * (ξ * t + α * t²/2))

        The α parameter controls the chirp rate (frequency sweep).

        Returns:
            Complex kernels of shape [num_scales, kernel_length]
        """
        t = torch.linspace(
            -self.kernel_length // 2,
            self.kernel_length // 2,
            self.kernel_length,
            device=self.delta_xi_raw.device,
        )  # [kernel_length]

        sigma = self.sigma.unsqueeze(1)  # [num_scales, 1]
        xi = self.xi.unsqueeze(1)  # [num_scales, 1]
        alpha = self.alpha.unsqueeze(1)  # [num_scales, 1]
        t = t.unsqueeze(0)  # [1, kernel_length]

        # Gaussian envelope
        gaussian = torch.exp(-(t**2) / (2 * sigma**2))

        # Chirped complex sinusoid
        phase = 2 * math.pi * (xi * t + alpha * t**2 / 2)
        sinusoid = torch.complex(torch.cos(phase), torch.sin(phase))

        kernel = gaussian * sinusoid  # [num_scales, kernel_length]

        # Normalize energy
        kernel = kernel / (kernel.abs().sum(dim=1, keepdim=True) + 1e-8)

        return kernel

    def _generate_morse_kernel(self) -> torch.Tensor:
        """
        Generate Generalized Morse Wavelet kernels.

        Morse wavelets are defined in the frequency domain:
        Ψ(ω) = U(ω) * a_{β,γ} * ω^β * exp(-ω^γ)

        where U(ω) is the Heaviside step function, and a_{β,γ} is a normalization constant.

        NOTE: Morse Wavelets provide asymmetric time-frequency localization through
        learnable γ (symmetry) and β (time-bandwidth) parameters.

        Returns:
            Complex kernels of shape [num_scales, kernel_length]
        """
        # Generate frequency axis
        freq = torch.fft.fftfreq(self.kernel_length, device=self.delta_xi_raw.device)
        freq = freq.unsqueeze(0)  # [1, kernel_length]

        # Map center frequencies to peak frequencies
        xi = self.xi.unsqueeze(1)  # [num_scales, 1]
        gamma = self.gamma.unsqueeze(1)  # [num_scales, 1]
        beta = self.beta.unsqueeze(1)  # [num_scales, 1]

        # Constrain gamma and beta to reasonable ranges
        gamma = F.softplus(gamma) + 1.0  # gamma >= 1
        beta = F.softplus(beta) + 1.0  # beta >= 1

        # Peak frequency for Morse wavelet
        omega_peak = (beta / gamma) ** (1 / gamma)

        # Scale frequency axis to align with desired center frequency
        # omega = freq * omega_peak / xi (normalized so peak aligns with xi)
        omega = freq.abs() * omega_peak / (xi + 1e-8)

        # Morse wavelet in frequency domain (positive frequencies only)
        psi_freq = (omega**beta) * torch.exp(-(omega**gamma))

        # Apply Heaviside: zero out negative frequencies
        psi_freq = psi_freq * (freq >= 0).float()

        # Normalize
        psi_freq = psi_freq / (psi_freq.sum(dim=1, keepdim=True) + 1e-8)

        # Convert to time domain via IFFT
        kernel = torch.fft.ifft(
            psi_freq.to(torch.complex64)
        )  # [num_scales, kernel_length]

        return kernel

    def _generate_kernels(self) -> torch.Tensor:
        """Generate complex convolution kernels based on filter_type."""
        if self.config.filter_type == "morlet":
            return self._generate_morlet_kernel()
        elif self.config.filter_type == "chirplet":
            return self._generate_chirplet_kernel()
        elif self.config.filter_type == "morse":
            return self._generate_morse_kernel()
        else:
            raise ValueError(f"Unknown filter_type: {self.config.filter_type}")

    def _fft_convolve(self, x: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
        """
        FFT-based complex convolution for large kernels.

        Uses rfft for real input and fft for complex kernel.
        Implements linear convolution via frequency-domain multiplication.

        Args:
            x: Input tensor [B, 1, L] (real, already padded)
            kernels: Complex kernels [num_scales, 1, kernel_length]

        Returns:
            Complex convolution output [B, num_scales, L]
        """
        B, _, L = x.shape
        num_scales = kernels.shape[0]
        kernel_length = kernels.shape[2]

        # FFT size for linear convolution: next_fast_len(L + kernel_length - 1)
        fft_size = int(2 ** math.ceil(math.log2(L + kernel_length - 1)))

        # Pad input and kernel to fft_size
        x_padded = torch.nn.functional.pad(
            x, (0, fft_size - L), mode="constant", value=0
        )
        kernel_padded = torch.nn.functional.pad(
            kernels, (0, fft_size - kernel_length), mode="constant", value=0
        )

        # Compute FFT: rfft for real input, fft for complex kernel
        X = torch.fft.rfft(x_padded, n=fft_size, dim=-1)  # [B, 1, N/2+1]
        K = torch.fft.fft(kernel_padded, n=fft_size, dim=-1)  # [num_scales, 1, N]

        # Prepare K for broadcasting with X: [num_scales, 1, N] -> [1, num_scales, N]
        K_expanded = K.permute(1, 0, 2)  # [1, num_scales, N]

        # Expand X to match K dimensions: [B, 1, N/2+1] -> [B, num_scales, N/2+1]
        X_expanded = X.expand(-1, num_scales, -1)  # [B, num_scales, N/2+1]

        # Pad K to match X size for element-wise multiplication
        # K is [num_scales, 1, N], X is [B, num_scales, N/2+1]
        # Expand K and zero-pad the frequency representation.
        K_expanded_full = torch.zeros(
            1, num_scales, fft_size, dtype=X.dtype, device=x.device
        )
        K_expanded_full[:, :, :fft_size] = K_expanded[:, :, :]

        # Frequency-domain multiplication (complex * complex)
        Y = X_expanded * K_expanded_full[:, :, : X_expanded.shape[-1]]

        # Inverse FFT to time domain (full complex result)
        y_complex = torch.fft.ifft(Y, n=fft_size, dim=-1)  # [B, num_scales, fft_size]

        # Crop to original length L
        out = y_complex[..., :L]  # [B, num_scales, L]

        return out

    def get_coverage_loss(self) -> torch.Tensor:
        """
        Compute Littlewood-Paley (Parseval Frame) regularization loss.

        This implements the frame condition: sum_i |Ψ_i(f)|² ≈ 1 for all f.

        The loss enforces:
        - Uniform frequency coverage (no gaps)
        - No excessive overlap (energy conservation)

        For Mel-scale filter banks, the target coverage curve is adapted to match
        the expected filter density in Mel space (dense at low freq, sparse at high freq).

        NOTE: Coverage Loss is essential for guaranteeing energy conservation and
        feature completeness (Frame Theory).

        Returns:
            Scalar loss value: ||S(f) - target||² for positive frequencies.
        """
        # Get time-domain kernels
        kernels = self._generate_kernels()  # [num_scales, kernel_length]

        # FFT to frequency domain
        K_freq = torch.fft.fft(kernels, dim=1)  # [num_scales, kernel_length]

        # Compute energy spectrum: |K_i(f)|²
        energy = (K_freq.abs()) ** 2  # [num_scales, kernel_length]

        # Sum over all scales: S(f) = sum_i |Ψ_i(f)|²
        S_f = energy.sum(dim=0)  # [kernel_length]

        # Only compute loss for positive frequencies (first half excluding DC)
        n_pos = self.kernel_length // 2
        S_f_pos = S_f[1 : n_pos + 1]

        # ================== Adaptive target based on scale_type ==================
        if self.config.scale_type == "mel":
            # For Mel scale: target follows Mel density curve (low-freq high, high-freq low)
            # 1. Generate linear frequency axis (0 to 0.5 normalized frequency)
            freqs = torch.linspace(0, 0.5, n_pos + 1, device=S_f_pos.device)[1:]

            # 2. Compute Mel-scale theoretical density distribution
            # Assume sample rate = 16000 (consistent with _init_center_frequencies)
            f_hz = freqs * 16000.0
            # Density proportional to derivative of Mel formula
            mel_density = 1.0 / (1.0 + f_hz / 700.0)

            # 3. Normalize both actual energy and target density by mean
            S_f_pos = S_f_pos / (S_f_pos.mean() + 1e-8)
            target = mel_density / (mel_density.mean() + 1e-8)

            log_S = torch.log1p(S_f_pos)
            log_target = torch.log1p(target)
            coverage_loss = F.mse_loss(log_S, log_target)
        else:
            # For linear scale: target is uniform
            S_f_pos = S_f_pos / (S_f_pos.mean() + 1e-8)
            target = torch.ones_like(S_f_pos)

            # MSE Loss
            coverage_loss = F.mse_loss(S_f_pos, target)

        return coverage_loss

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply physical filter bank to input signal.

        Args:
            x: Input tensor of shape [Batch, C_in, Length]

        Returns:
            Output tensor of shape [Batch, C_in * num_scales, Length]
        """
        B, C, L = x.shape
        assert C == self.c_in, f"Expected {self.c_in} channels, got {C}"

        # Generate complex kernels: [num_scales, kernel_length]
        kernels = self._generate_kernels()

        # Reshape for conv1d: [num_scales, 1, kernel_length]
        kernels = kernels.unsqueeze(1)

        # Reshape input for broadcasting: [B * C_in, 1, L]
        x_reshaped = x.view(B * C, 1, L)

        # Ensure input is float
        x_reshaped = x_reshaped.float()

        # Use FFT convolution for large kernels (O(L log L) vs O(L * K))
        use_fft = self.kernel_length >= 128

        if use_fft:
            # FFT-based convolution path
            # Pre-pad with replicate padding
            padding = self.kernel_length // 2
            x_padded = F.pad(x_reshaped, (padding, padding), mode="replicate")

            # Compute complex convolution via FFT
            out_complex = self._fft_convolve(x_padded, kernels)  # [B*C, num_scales, L]

            # Compute modulus from complex result
            out_mag = torch.sqrt(out_complex.real**2 + out_complex.imag**2 + 1e-8)
        else:
            # Direct F.conv1d path (faster for small kernels)
            kernel_real = kernels.real  # [num_scales, 1, kernel_length]
            kernel_imag = kernels.imag  # [num_scales, 1, kernel_length]

            padding = self.kernel_length // 2
            x_padded = F.pad(x_reshaped, (padding, padding), mode="replicate")
            out_real = F.conv1d(x_padded, kernel_real.float(), padding=0)
            out_imag = F.conv1d(x_padded, kernel_imag.float(), padding=0)

            # Modulus: sqrt(Re² + Im²)
            out_mag = torch.sqrt(out_real**2 + out_imag**2 + 1e-8)

        # Adjust output length to match input
        L_out = out_mag.shape[-1]
        if L_out > L:
            # Trim to match input length
            start = (L_out - L) // 2
            out_mag = out_mag[:, :, start : start + L]
        elif L_out < L:
            # Pad to match input length
            pad_total = L - L_out
            pad_left = pad_total // 2
            pad_right = pad_total - pad_left
            out_mag = F.pad(out_mag, (pad_left, pad_right))

        # Reshape to [B, C_in * num_scales, L]
        out_mag = out_mag.view(B, C, self.num_scales, L)
        out_mag = out_mag.view(B, C * self.num_scales, L)

        return out_mag


# ==============================================================================
# 3. Core Module: FactorizedMixing
# ==============================================================================


class FactorizedMixing(nn.Module):
    """
    Factorized channel mixing layer with preserved physical structure.

    This module implements two-step mixing:
    1. Intra-subcarrier mixing (frequency-domain): Mix scales within each subcarrier
       using grouped convolution (groups = mixing_groups or c_in).
    2. GELU activation (configurable).
    3. Inter-subcarrier mixing (spatial-domain): Mix information across subcarriers.

    Both convolution layers are constrained with Spectral Normalization to limit
    Lipschitz constant and ensure stability.

    Args:
        in_channels: Total input channels (C_in * num_scales).
        c_in: Original number of subcarrier/sensor channels.
        num_scales: Number of scales per subcarrier.
        intra_out_per_group: Output channels per group for intra-subcarrier mixing.
        inter_out: Output channels for inter-subcarrier mixing.
        activation: Activation function between steps ('gelu', 'relu', 'none').
        mixing_groups: Number of groups for intra-convolution.
            - If None (default), uses c_in for subcarrier isolation.
            - If set to a specific value, uses that value for groups.
            - For Stage 1 (c_in=physical subcarriers): use c_in or None.
            - For Stage 2+ (c_in=intermediate channels): use smaller value like 8, 16, 32.
    """

    def __init__(
        self,
        in_channels: int,
        c_in: int,
        num_scales: int,
        intra_out_per_group: int,
        inter_out: int,
        activation: Literal["gelu", "relu", "none"] = "gelu",
        mixing_groups: Optional[int] = None,
    ):
        super().__init__()
        self.c_in = c_in
        self.in_channels = in_channels
        self.num_scales = num_scales
        # Use mixing_groups if provided, otherwise use c_in
        self.mixing_groups = mixing_groups if mixing_groups is not None else c_in

        # Validate mixing_groups
        assert in_channels % self.mixing_groups == 0, (
            f"in_channels ({in_channels}) must be divisible by mixing_groups ({self.mixing_groups})"
        )

        # Calculate scales per subcarrier
        assert in_channels % c_in == 0, (
            f"in_channels ({in_channels}) must be divisible by c_in ({c_in})"
        )
        self.scales_per_channel = in_channels // c_in

        # Step 1: Intra-subcarrier mixing (grouped 1x1 conv)
        # groups = mixing_groups (configurable, defaults to c_in)
        intra_total_out = self.mixing_groups * intra_out_per_group
        self.intra_conv = spectral_norm(
            nn.Conv1d(
                in_channels=in_channels,
                out_channels=intra_total_out,
                kernel_size=1,
                groups=self.mixing_groups,
                bias=True,
            )
        )

        # Activation between steps
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        else:
            self.act = nn.Identity()

        # Step 2: Inter-subcarrier mixing (standard 1x1 conv)
        self.inter_conv = spectral_norm(
            nn.Conv1d(
                in_channels=intra_total_out,
                out_channels=inter_out,
                kernel_size=1,
                bias=True,
            )
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, in_channels, Length]
               where in_channels = C_in * num_scales

        Returns:
            Output tensor of shape [Batch, inter_out, Length]
        """
        # Step 1: Intra-subcarrier mixing
        out = self.intra_conv(x)

        # Activation
        out = self.act(out)

        # Step 2: Inter-subcarrier mixing
        out = self.inter_conv(out)

        return out


# ==============================================================================
# 4. Core Module: LearnableBlurPool1d
# ==============================================================================


class LearnableBlurPool1d(nn.Module):
    """
    Learnable anti-aliasing blur pooling layer.

    Implements depthwise separable Gaussian low-pass filtering followed by
    downsampling. The blur sigma is learnable but constrained to be >= 0.8
    to ensure proper anti-aliasing.

    This prevents aliasing artifacts during downsampling by removing high
    frequencies before strided convolution.

    Args:
        channels: Number of input/output channels.
        stride: Downsampling factor.
        kernel_size: Size of the Gaussian blur kernel.
        min_sigma: Minimum sigma value for anti-aliasing (default: 0.8).
    """

    def __init__(
        self,
        channels: int,
        stride: int = 2,
        kernel_size: int = 5,
        min_sigma: float = 0.8,
        use_padding: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.stride = stride
        self.kernel_size = kernel_size
        self.min_sigma = min_sigma
        self.use_padding = use_padding

        # Learnable sigma (raw, will be constrained)
        self.sigma_raw = nn.Parameter(torch.tensor(1.0))

        # Create kernel grid (fixed)
        self.register_buffer(
            "kernel_grid",
            torch.linspace(-(kernel_size // 2), kernel_size // 2, kernel_size),
        )

    @property
    def sigma(self) -> torch.Tensor:
        """Constrained sigma parameter (>= min_sigma for anti-aliasing)."""
        return F.softplus(self.sigma_raw) + self.min_sigma

    def _get_gaussian_kernel(self) -> torch.Tensor:
        """Generate normalized 1D Gaussian kernel."""
        sigma = self.sigma
        kernel = torch.exp(-(self.kernel_grid**2) / (2 * sigma**2))
        kernel = kernel / kernel.sum()  # Normalize
        return kernel

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape [Batch, Channels, Length]

        Returns:
            Blurred and downsampled tensor of shape [Batch, Channels, Length // stride]
        """
        B, C, L = x.shape

        # Get Gaussian kernel
        kernel = self._get_gaussian_kernel()  # [kernel_size]

        # Reshape for depthwise conv: [channels, 1, kernel_size]
        kernel = kernel.view(1, 1, -1).expand(C, 1, -1)

        # Apply depthwise blur with replicate padding
        if self.use_padding:
            pad = self.kernel_size // 2
            x_padded = F.pad(x, (pad, pad), mode="replicate")
            out = F.conv1d(x_padded, kernel, stride=self.stride, padding=0, groups=C)
        else:
            out = F.conv1d(x, kernel, stride=self.stride, groups=C)

        return out


# ==============================================================================
# 5. Composite Module: LMSBlock
# ==============================================================================


# ==============================================================================
# 5. Composite Module: LMSBlock (Scattering Architecture)
# ==============================================================================


class LMSBlock(nn.Module):
    """
    LMS Block with Hierarchical Scattering Architecture (Andén & Mallat, 2014).

    Dual-Path Design:
        Path A (u_next): High-frequency stream preserving time resolution -> next layer
        Path B (s_out): Low-frequency invariant features via blur pooling -> aggregation

    Input: [B, C_in, L] -> Output: (u_next=[B, inter_out, L], s_out=[B, inter_out, L/stride])

    Key Insight: Serial architectures lose high-frequency modulation info during downsampling.
    Dual-path preserves high-frequency stream while extracting translation-invariant features.
    """

    def __init__(
        self,
        c_in: int,
        filter_config: FilterConfig,
        intra_out_per_group: int,
        inter_out: int,
        blur_stride: int = 2,
        blur_kernel_size: int = 5,
        blur_min_sigma: float = 0.8,
        use_log: bool = True,
        mixing_activation: Literal["gelu", "relu", "none"] = "gelu",
        use_residual: bool = False,
        eps: float = 1e-6,
        blur_use_padding: bool = True,
        is_last_block: bool = False,
    ):
        super().__init__()
        self.c_in = c_in
        self.use_log = use_log
        self.use_residual = use_residual
        self.eps = eps
        self.is_last_block = is_last_block

        self.filter_bank = PhysicalFilterBank(filter_config, c_in)
        filter_out_channels = c_in * filter_config.num_scales

        # Check if use_joint_mixing is enabled in filter_config
        use_joint = getattr(filter_config, "use_joint_mixing", False)

        if use_joint:
            # Use JointFactorizedMixing for Joint Time-Frequency Scattering
            # This applies 2D convolution on the time-frequency representation
            self.mixing = JointFactorizedMixing(
                in_channels=filter_out_channels,
                c_in=c_in,
                num_scales=filter_config.num_scales,
                intra_out_per_group=intra_out_per_group,
                inter_out=inter_out,
                activation=mixing_activation,
                freq_kernel_size=getattr(filter_config, "freq_kernel_size", 3),
                time_kernel_size=getattr(filter_config, "time_kernel_size", 15),
                reduce_scales=getattr(filter_config, "reduce_scales", True),
                pool_type=getattr(filter_config, "pool_type", "learnable"),
                mixing_groups=getattr(filter_config, "mixing_groups", None),
            )
        else:
            # Use standard FactorizedMixing
            self.mixing = FactorizedMixing(
                in_channels=filter_out_channels,
                c_in=c_in,
                num_scales=filter_config.num_scales,
                intra_out_per_group=intra_out_per_group,
                inter_out=inter_out,
                activation=mixing_activation,
                mixing_groups=getattr(filter_config, "mixing_groups", None),
            )
        self.blur_pool = LearnableBlurPool1d(
            channels=inter_out,
            stride=blur_stride,
            kernel_size=blur_kernel_size,
            min_sigma=blur_min_sigma,
            use_padding=blur_use_padding,
        )

        self.residual_proj = None
        if use_residual:
            if c_in == inter_out and blur_stride == 1:
                self.residual_proj = nn.Identity()
            elif blur_stride > 1:
                self.residual_proj = nn.Sequential(
                    nn.AvgPool1d(blur_stride),
                    nn.Conv1d(c_in, inter_out, 1)
                    if c_in != inter_out
                    else nn.Identity(),
                )
            else:
                self.residual_proj = nn.Conv1d(c_in, inter_out, 1)

    def forward(self, x: torch.Tensor):
        """
        Dual-path forward pass.

        Returns:
            u_next: High-frequency stream for next layer [B, inter_out, L]
            s_out: Translation-invariant features [B, inter_out, L // blur_stride]
        """
        identity = x

        # Wavelet scattering: [B, C_in, L] -> [B, C_in * num_scales, L]
        out = self.filter_bank(x)

        # Log compression for stability
        if self.use_log:
            out = torch.log1p(out / self.eps)

        # Channel mixing: [B, C_in * num_scales, L] -> [B, inter_out, L]
        out = self.mixing(out)

        # Preserve full-resolution mixed features for the next stage.
        if self.is_last_block:
            u_next = None
        else:
            u_next = out

        # Blur pooling only applies to the invariant branch.
        s_out = self.blur_pool(out)

        # Optional residual connection on s_out
        if self.use_residual and self.residual_proj is not None:
            identity_proj = self.residual_proj(identity)
            if s_out.shape[-1] != identity_proj.shape[-1]:
                min_len = min(s_out.shape[-1], identity_proj.shape[-1])
                s_out = s_out[:, :, :min_len]
                identity_proj = identity_proj[:, :, :min_len]
            s_out = s_out + identity_proj

        return u_next, s_out


# ==============================================================================
# 6. Main Architecture: LMSNet
# ==============================================================================


# ==============================================================================
# 6. Main Architecture: LMSNet (Hierarchical Scattering Network)
# ==============================================================================


class LMSNet(nn.Module):
    """
    LMS-Net: Learnable Mel-Scattering Network with Hierarchical Scattering Architecture.

    Architecture: Input -> [LMSBlock x N] -> Concatenate(s_out) -> Pool -> MLP -> Logits

    Multi-scale invariant features are concatenated in the channel dimension:
        - s_out_1: Short-timescale invariance (fast variations)
        - s_out_2: Medium-timescale invariance (rhythmic patterns)
        - s_out_3: Long-timescale invariance (global trends)

    Features:
        - Lipschitz stability via physical scattering
        - Parseval frame regularization via coverage loss
        - Multi-scale invariant feature aggregation
    """

    def __init__(
        self,
        c_in: int,
        input_length: int,
        num_classes: int,
        stage_configs: Optional[
            List[Tuple[FilterConfig, int, int, int, int, float]]
        ] = None,
        use_log: bool = True,
        mixing_activation: Literal["gelu", "relu", "none"] = "gelu",
        mlp_hidden: int = 256,
        dropout: float = 0.1,
        use_std: bool = False,
        align_time: bool = True,
        blur_use_padding: bool = True,
    ):
        super().__init__()
        self.c_in = c_in
        self.input_length = input_length
        self.num_classes = num_classes
        self.align_time = align_time

        if stage_configs is None:
            stage_configs = [
                (FilterConfig(filter_type="morlet", num_scales=4), 8, 128, 2, 5, 0.8),
                (FilterConfig(filter_type="morlet", num_scales=4), 8, 256, 2, 5, 0.8),
                (FilterConfig(filter_type="morlet", num_scales=4), 8, 512, 2, 5, 0.8),
            ]

        self.blocks = nn.ModuleList()
        current_channels = c_in

        self.inter_outs = []
        self.blur_strides = []

        for (
            filter_config,
            intra_out,
            inter_out,
            blur_stride,
            blur_kernel_size,
            blur_min_sigma,
        ) in stage_configs:
            block = LMSBlock(
                c_in=current_channels,
                filter_config=filter_config,
                intra_out_per_group=intra_out,
                inter_out=inter_out,
                blur_stride=blur_stride,
                blur_kernel_size=blur_kernel_size,
                blur_min_sigma=blur_min_sigma,
                use_log=use_log,
                mixing_activation=mixing_activation,
                blur_use_padding=blur_use_padding,
            )
            self.blocks.append(block)
            # The dual-path block forwards ``u_next`` with ``inter_out``
            # channels (not ``c_in * num_scales``), so this is the input width
            # of the next stage.
            current_channels = inter_out
            self.inter_outs.append(inter_out)
            self.blur_strides.append(blur_stride)

        self.blur_use_padding = blur_use_padding

        if blur_use_padding:
            self.use_std = use_std
            factor = 2 if self.use_std else 1
        else:
            factor = ((input_length - blur_kernel_size) // blur_stride) + 1

        self.pool_dim = sum(self.inter_outs) * factor

        # Simplified head to reduce overfitting: single hidden layer
        self.head = nn.Sequential(
            # --- 特征融合层 ---
            nn.Linear(self.pool_dim, mlp_hidden),
            nn.BatchNorm1d(mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            # --- 分类输出层 ---
            nn.Linear(mlp_hidden, num_classes),
        )

    def get_total_coverage_loss(self) -> torch.Tensor:
        """Aggregate Parseval frame regularization loss from all filter banks."""
        total_loss = torch.tensor(0.0, device=next(self.parameters()).device)
        for block in self.blocks:
            total_loss = total_loss + block.filter_bank.get_coverage_loss()
        return total_loss

    def _align_features(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        """Align s_out features to same time length via interpolation."""
        if not self.align_time or len(features) <= 1:
            return features

        lengths = [f.shape[-1] for f in features]
        target_length = max(lengths)

        aligned_features = []
        for feat, length in zip(features, lengths):
            if length == target_length:
                aligned_features.append(feat)
            else:
                aligned = F.interpolate(
                    feat,
                    size=target_length,
                    mode="linear",
                    align_corners=True,
                )
                aligned_features.append(aligned)

        return aligned_features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Hierarchical scattering forward pass.

        Returns: Logits [B, num_classes]
        """
        # Collect invariant features from each scale
        s_out_list = []
        u_current = x

        for block in self.blocks:
            u_current, s_out = block(u_current)
            s_out_list.append(s_out)

        # Align and concatenate in channel dimension
        if self.align_time:
            s_out_aligned = self._align_features(s_out_list)
        else:
            s_out_aligned = s_out_list

        s_out_cat = torch.cat(s_out_aligned, dim=1)

        if self.blur_use_padding:
            # Global pooling
            mean_features = s_out_cat.mean(dim=-1)

            if self.use_std:
                std_features = s_out_cat.std(dim=-1)
                pooled = torch.cat([mean_features, std_features], dim=-1)
            else:
                pooled = mean_features
        else:
            pooled = s_out_cat.flatten(start_dim=1)

        return self.head(pooled)


# ==============================================================================
# Convenience function for building LMSNet
# ==============================================================================


def build_lms_net(
    c_in: int,
    input_length: int,
    num_classes: int,
    filter_type: Literal["morlet", "chirplet", "morse"] = "morlet",
    scale_type: Literal["mel", "linear"] = "mel",
    num_scales: int = 4,
    num_stages: int = 3,
    base_channels: int = 128,
    use_log: bool = True,
    **kwargs,
) -> LMSNet:
    """
    Convenience function to build LMSNet with common configurations.

    Args:
        c_in: Number of input channels.
        input_length: Input sequence length.
        num_classes: Number of output classes.
        filter_type: Type of physical filter ('morlet', 'chirplet', 'morse').
        scale_type: Scale spacing ('mel', 'linear').
        num_scales: Number of scales per filter bank.
        num_stages: Number of LMSBlock stages.
        base_channels: Base channel count (doubles each stage).
        use_log: Whether to use log compression.
        **kwargs: Additional arguments passed to LMSNet.

    Returns:
        Configured LMSNet instance.
    """
    # Build stage configurations with increasing channels
    stage_configs = []
    for i in range(num_stages):
        channels = base_channels * (2**i)
        filter_config = FilterConfig(
            filter_type=filter_type,
            scale_type=scale_type,
            num_scales=num_scales,
        )
        # (filter_config, intra_out_per_group, inter_out, blur_stride, blur_kernel_size, blur_min_sigma)
        stage_configs.append((filter_config, num_scales * 2, channels, 2, 5, 0.8))

    return LMSNet(
        c_in=c_in,
        input_length=input_length,
        num_classes=num_classes,
        stage_configs=stage_configs,
        use_log=use_log,
        **kwargs,
    )
