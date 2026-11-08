"""
Time warping transformation for [B, C, T] tensors using PyTorch.

Provides time deformation operations that apply non-linear time distortions
to time series data while preserving endpoints.
"""

import torch
import torch.nn.functional as F


def apply_time_deformation_1d(
    x: torch.Tensor,
    warp_strength: float = 0.5,
    shift_ratio: float = 0.0,
    num_knots: int = 6,
) -> torch.Tensor:
    """
    Apply time deformation to a 1D tensor of shape [T].

    Args:
        x: Input tensor of shape [T]
        warp_strength: Strength of time warping, must be < 1
        shift_ratio: Ratio of random shift
        num_knots: Number of knots for cubic spline interpolation

    Returns:
        Warped tensor of shape [T]
    """
    assert warp_strength < 1, f"warp_strength must be < 1, got {warp_strength}"

    T = x.shape[0]
    device = x.device
    dtype = x.dtype
    t = torch.arange(T, device=device, dtype=dtype)

    if warp_strength > 0:
        knot_t = torch.linspace(0, T - 1, num_knots, device=device, dtype=dtype)
        disp = torch.empty(num_knots, device=device, dtype=dtype).uniform_(-1, 1)
        disp[0] = 0
        disp[-1] = 0

        tau = _cubic_spline_interp(knot_t, disp, t)

        grad = tau[1:] - tau[:-1]
        max_grad = torch.abs(grad).max()

        if max_grad > 1e-6:
            tau = tau * warp_strength / max_grad
    else:
        tau = torch.zeros(T, device=device, dtype=dtype)

    if shift_ratio > 0:
        shift = (
            torch.empty(1, device=device, dtype=dtype).uniform_(
                -shift_ratio, shift_ratio
            )
            * T
        )
        tau = tau + shift

    t_new = (t - tau).clamp(0, T - 1)
    return _linear_interp_1d(x, t_new)


def apply_time_deformation(
    x: torch.Tensor,
    warp_strength: float = 0.5,
    shift_ratio: float = 0.0,
    num_knots: int = 6,
    same_warp: bool = False,
    cache: dict | None = None,
) -> torch.Tensor:
    """
    Apply time deformation to a tensor of shape [B, C, T].

    Args:
        x: Input tensor of shape [B, C, T]
        warp_strength: Strength of time warping, must be < 1
        shift_ratio: Ratio of random shift
        num_knots: Number of knots for cubic spline interpolation
        same_warp: If True, apply same warp to all samples in batch

    Returns:
        Warped tensor of shape [B, C, T]
    """
    assert warp_strength < 1, f"warp_strength must be < 1, got {warp_strength}"
    assert x.dim() == 3, f"Expected input shape [B, C, T], got {x.shape}"

    B, _, T = x.shape
    device = x.device
    dtype = x.dtype

    if same_warp:
        t_new = _generate_warp_indices(
            T, warp_strength, shift_ratio, num_knots, device, dtype
        )
        t_new = t_new.unsqueeze(0).expand(B, -1)
    else:
        t_new = _generate_warp_indices_batch(
            B, T, warp_strength, shift_ratio, num_knots, device, dtype
        )

    return _batch_linear_interp(x, t_new, cache=cache)


def _generate_warp_indices(
    T: int,
    warp_strength: float,
    shift_ratio: float,
    num_knots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Generate warped time indices for a single sequence."""
    t = torch.arange(T, device=device, dtype=dtype)

    if warp_strength > 0:
        knot_t = torch.linspace(0, T - 1, num_knots, device=device, dtype=dtype)
        disp = torch.empty(num_knots, device=device, dtype=dtype).uniform_(-1, 1)
        disp[0] = 0
        disp[-1] = 0

        tau = _cubic_spline_interp(knot_t, disp, t)

        grad = tau[1:] - tau[:-1]
        max_grad = torch.abs(grad).max()

        if max_grad > 1e-6:
            tau = tau * warp_strength / max_grad
    else:
        tau = torch.zeros(T, device=device, dtype=dtype)

    if shift_ratio > 0:
        shift = (
            torch.empty(1, device=device, dtype=dtype).uniform_(
                -shift_ratio, shift_ratio
            )
            * T
        )
        tau = tau + shift

    return (t - tau).clamp(0, T - 1)


def _generate_warp_indices_batch(
    B: int,
    T: int,
    warp_strength: float,
    shift_ratio: float,
    num_knots: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    t = torch.arange(T, device=device, dtype=dtype)

    if warp_strength > 0:
        knot_t = torch.linspace(0, T - 1, num_knots, device=device, dtype=dtype)
        disp = torch.empty(B, num_knots, device=device, dtype=dtype).uniform_(-1, 1)
        disp[:, 0] = 0
        disp[:, -1] = 0

        tau = _cubic_spline_interp_batch(knot_t, disp, t)

        grad = tau[:, 1:] - tau[:, :-1]
        max_grad = torch.abs(grad).amax(dim=1, keepdim=True)
        scale = torch.ones_like(max_grad)
        valid = max_grad > 1e-6
        scale[valid] = warp_strength / max_grad[valid]
        tau = tau * scale
    else:
        tau = torch.zeros(B, T, device=device, dtype=dtype)

    if shift_ratio > 0:
        shift = (
            torch.empty(B, 1, device=device, dtype=dtype).uniform_(
                -shift_ratio, shift_ratio
            )
            * T
        )
        tau = tau + shift

    return (t.unsqueeze(0) - tau).clamp(0, T - 1)


def _cubic_spline_interp(
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    """
    Perform natural cubic spline interpolation.

    Uses not-a-knot end conditions.
    """
    device = x_knots.device
    dtype = x_knots.dtype
    n = len(x_knots)

    h = x_knots[1:] - x_knots[:-1]
    alpha = torch.zeros(n, device=device, dtype=dtype)

    for i in range(1, n - 1):
        alpha[i] = (3 / h[i]) * (y_knots[i + 1] - y_knots[i]) - (3 / h[i - 1]) * (
            y_knots[i] - y_knots[i - 1]
        )

    l = torch.ones(n, device=device, dtype=dtype)
    mu = torch.zeros(n, device=device, dtype=dtype)
    z = torch.zeros(n, device=device, dtype=dtype)

    for i in range(1, n - 1):
        l[i] = 2 * (x_knots[i + 1] - x_knots[i - 1]) - h[i - 1] * mu[i - 1]
        mu[i] = h[i] / l[i]
        z[i] = (alpha[i] - h[i - 1] * z[i - 1]) / l[i]

    M = torch.zeros(n, device=device, dtype=dtype)
    for i in range(n - 2, -1, -1):
        M[i] = z[i] - mu[i] * M[i + 1]

    y_query = torch.zeros_like(x_query)
    indices = torch.searchsorted(x_knots[:-1].contiguous(), x_query) - 1
    indices = indices.clamp(0, n - 2)

    for i in range(n - 1):
        mask = indices == i
        if not mask.any():
            continue

        x_q = x_query[mask]
        h_i = h[i]

        a = (M[i + 1] - M[i]) / (6 * h_i)
        b = M[i] / 2
        c = (y_knots[i + 1] - y_knots[i]) / h_i - (M[i + 1] + 2 * M[i]) * h_i / 6
        d = y_knots[i]

        dx = x_q - x_knots[i]
        y_query[mask] = a * dx**3 + b * dx**2 + c * dx + d

    return y_query


def _cubic_spline_interp_batch(
    x_knots: torch.Tensor,
    y_knots: torch.Tensor,
    x_query: torch.Tensor,
) -> torch.Tensor:
    device = x_knots.device
    dtype = x_knots.dtype
    B, n = y_knots.shape

    h = x_knots[1:] - x_knots[:-1]
    alpha = torch.zeros(B, n, device=device, dtype=dtype)
    alpha[:, 1:-1] = (3.0 / h[1:]).unsqueeze(0) * (
        y_knots[:, 2:] - y_knots[:, 1:-1]
    ) - (3.0 / h[:-1]).unsqueeze(0) * (y_knots[:, 1:-1] - y_knots[:, :-2])

    l = torch.ones(B, n, device=device, dtype=dtype)
    mu = torch.zeros(B, n, device=device, dtype=dtype)
    z = torch.zeros(B, n, device=device, dtype=dtype)

    for i in range(1, n - 1):
        l[:, i] = 2 * (x_knots[i + 1] - x_knots[i - 1]) - h[i - 1] * mu[:, i - 1]
        mu[:, i] = h[i] / l[:, i]
        z[:, i] = (alpha[:, i] - h[i - 1] * z[:, i - 1]) / l[:, i]

    M = torch.zeros(B, n, device=device, dtype=dtype)
    for i in range(n - 2, -1, -1):
        M[:, i] = z[:, i] - mu[:, i] * M[:, i + 1]

    y_query = torch.zeros(B, x_query.shape[0], device=device, dtype=dtype)
    indices = torch.searchsorted(x_knots[:-1].contiguous(), x_query) - 1
    indices = indices.clamp(0, n - 2)

    for i in range(n - 1):
        mask = indices == i
        if not mask.any():
            continue

        x_q = x_query[mask]
        h_i = h[i]

        a = (M[:, i + 1] - M[:, i]) / (6 * h_i)
        b = M[:, i] / 2
        c = (y_knots[:, i + 1] - y_knots[:, i]) / h_i - (
            M[:, i + 1] + 2 * M[:, i]
        ) * h_i / 6
        d = y_knots[:, i]

        dx = x_q.unsqueeze(0) - x_knots[i]
        y_query[:, mask] = (
            a.unsqueeze(1) * dx**3
            + b.unsqueeze(1) * dx**2
            + c.unsqueeze(1) * dx
            + d.unsqueeze(1)
        )

    return y_query


def _linear_interp_1d(x: torch.Tensor, t_new: torch.Tensor) -> torch.Tensor:
    """Linear interpolation for 1D tensor."""
    T = x.shape[0]
    t_floor = t_new.floor().long()
    t_ceil = t_floor + 1

    t_floor = t_floor.clamp(0, T - 1)
    t_ceil = t_ceil.clamp(0, T - 1)

    weight = t_new - t_floor.float()

    return x[t_floor] * (1 - weight) + x[t_ceil] * weight


def _batch_linear_interp(
    x: torch.Tensor,
    t_new: torch.Tensor,
    cache: dict | None = None,
) -> torch.Tensor:
    """
    Batched linear interpolation using grid_sample.

    Args:
        x: Input tensor of shape [B, C, T]
        t_new: New time indices of shape [B, T], values in range [0, T-1]

    Returns:
        Interpolated tensor of shape [B, C, T]
    """
    B, _, T = x.shape
    device = x.device
    dtype = x.dtype

    t_norm = 2.0 * t_new / (T - 1) - 1.0

    grid = None
    if cache is not None:
        grid = cache.get("grid")

    if (
        grid is None
        or grid.shape != (B, 1, T, 2)
        or grid.device != device
        or grid.dtype != dtype
    ):
        grid = torch.empty(B, 1, T, 2, device=device, dtype=dtype)
        if cache is not None:
            cache["grid"] = grid

    grid[:, 0, :, 0] = t_norm
    grid[:, 0, :, 1] = 0

    x_4d = x.unsqueeze(2)

    x_warped = F.grid_sample(
        x_4d, grid, mode="bilinear", padding_mode="border", align_corners=True
    )

    return x_warped.squeeze(2)


class TimeWarpTransform:
    """
    Time warping transform for data augmentation.

    Example:
        >>> transform = TimeWarpTransform(warp_strength=0.5)
        >>> x = torch.randn(4, 3, 100)
        >>> x_warped = transform(x)
    """

    def __init__(
        self,
        warp_strength: float = 0.5,
        shift_ratio: float = 0.0,
        num_knots: int = 6,
        same_warp: bool = False,
    ):
        if warp_strength >= 1:
            raise ValueError(f"warp_strength must be < 1, got {warp_strength}")

        self.warp_strength: float = warp_strength
        self.shift_ratio: float = shift_ratio
        self.num_knots: int = num_knots
        self.same_warp: bool = same_warp
        self._cache: dict[str, torch.Tensor] = {}

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return apply_time_deformation(
            x,
            warp_strength=self.warp_strength,
            shift_ratio=self.shift_ratio,
            num_knots=self.num_knots,
            same_warp=self.same_warp,
            cache=self._cache,
        )
