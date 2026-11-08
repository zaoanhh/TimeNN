"""Small matplotlib helpers shared by the figure scripts.

The functions provide Nature-style sizes, an accessible colour palette and
bar/line/heatmap wrappers.  They are intentionally dependency-free beyond
matplotlib and numpy.
"""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, Literal, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Colormap, LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.figure import Figure


FigureSize = Literal["single", "double", "square", "wide", "heatmap"]
PathLike = str | Path

MM_PER_INCH = 25.4
NATURE_SINGLE_COLUMN_WIDTH_MM = 45.0
NATURE_DOUBLE_COLUMN_WIDTH_MM =91.0
NATURE_MAX_HEIGHT_MM = 85.0


def mm_to_inch(value_mm: float) -> float:
    return value_mm / MM_PER_INCH


factors = 1.5
FIGURE_SIZE_PRESETS: dict[FigureSize, tuple[float, float]] = {
    "single": (
        mm_to_inch(NATURE_SINGLE_COLUMN_WIDTH_MM * factors),
        mm_to_inch(30.0 * factors),
    ),
    "double": (
        mm_to_inch(NATURE_DOUBLE_COLUMN_WIDTH_MM * factors),
        mm_to_inch(35.0 * factors),
    ),
    "square": (
        mm_to_inch(NATURE_SINGLE_COLUMN_WIDTH_MM * factors),
        mm_to_inch(45.0 * factors),
    ),
    "wide": (
        mm_to_inch(NATURE_DOUBLE_COLUMN_WIDTH_MM * factors),
        mm_to_inch(30.0 * factors),
    ),
    "heatmap": (
        mm_to_inch(NATURE_SINGLE_COLUMN_WIDTH_MM * factors),
        mm_to_inch(36.0 * factors),
    ),
}

# Official accessible palette shown in the Nature research figure guide.
NATURE_PALETTE: dict[str, str] = {
    "black": "#000000",
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# Yellow is kept available but excluded from the default cycle because it loses
# contrast on white backgrounds in line and bar charts.
NATURE_COLOR_CYCLE: tuple[str, ...] = (
    NATURE_PALETTE["black"],
    NATURE_PALETTE["blue"],
    NATURE_PALETTE["vermillion"],
    NATURE_PALETTE["bluish_green"],
    NATURE_PALETTE["sky_blue"],
    NATURE_PALETTE["orange"],
    NATURE_PALETTE["reddish_purple"],
)

# Nature recommends relying on accessible colours first and avoiding decorative
# fills. Keep hatching sparse and optional for grayscale redundancy only.
NATURE_HATCH_PATTERNS: tuple[str, ...] = ("", "/", "\\", "|", "-", "+")
NATURE_MARKERS: tuple[str, ...] = ("o", "s", "^", "D", "v", "P", "X")
NATURE_LINESTYLES: tuple[str, ...] = ("-", "--", "-.", ":")

NATURE_SEQUENTIAL_CMAP = LinearSegmentedColormap.from_list(
    "nature_sequential",
    [
        "#FFFFFF",
        NATURE_PALETTE["sky_blue"],
        NATURE_PALETTE["blue"],
        NATURE_PALETTE["black"],
    ],
)
NATURE_DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "nature_diverging",
    [NATURE_PALETTE["vermillion"], "#F7F7F7", NATURE_PALETTE["blue"]],
)

DEFAULT_DPI = 300
DEFAULT_FIGURE_DPI = 150
DEFAULT_FONT_SIZE_PT = 6.0
DEFAULT_AXES_LINEWIDTH = 0.8
DEFAULT_LINE_WIDTH = 1.2
DEFAULT_GRID_LINEWIDTH = 0.4
DEFAULT_MARKER_SIZE = 4.0


def _validate_font_size(font_size: float) -> None:
    if not 5.0 <= font_size <= 7.0:
        raise ValueError(
            "Nature figure text should typically remain between 5pt and 7pt."
        )


def resolve_figsize(
    figsize: FigureSize | Sequence[float] | None,
) -> tuple[float, float]:
    if figsize is None:
        return FIGURE_SIZE_PRESETS["single"]

    if isinstance(figsize, str):
        if figsize not in FIGURE_SIZE_PRESETS:
            raise ValueError(f"Unknown figsize preset: {figsize}")
        return FIGURE_SIZE_PRESETS[figsize]

    if len(figsize) != 2:
        raise ValueError("figsize must contain exactly two values.")

    width, height = float(figsize[0]), float(figsize[1])
    if width <= 0 or height <= 0:
        raise ValueError("figsize values must be positive.")
    return width, height


def _normalize_series_count(
    values: Sequence[float] | Sequence[Sequence[float]],
    reference_length: int | None = None,
) -> list[np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [array]
    if array.ndim != 2:
        raise ValueError("Expected a 1D or 2D numeric array.")

    if reference_length is None:
        return [row for row in array]

    if array.shape[1] == reference_length:
        return [row for row in array]
    if array.shape[0] == reference_length:
        return [array[:, idx] for idx in range(array.shape[1])]
    raise ValueError("Could not align the 2D data with the provided reference length.")


def _normalize_box_data(values: Sequence[Sequence[float]]) -> list[np.ndarray]:
    if len(values) == 0:
        raise ValueError("Boxplot data cannot be empty.")
    box_data: list[np.ndarray] = []
    for series in values:
        series_array = np.asarray(series, dtype=float)
        if series_array.ndim != 1:
            raise ValueError("Each boxplot series must be 1D.")
        if series_array.size == 0:
            raise ValueError("Boxplot series cannot be empty.")
        box_data.append(series_array)
    return box_data


def _broadcast_style_values(
    values: Sequence[Any] | None,
    count: int,
    defaults: Sequence[Any],
) -> list[Any]:
    if count <= 0:
        return []

    if values is None:
        source = defaults
    else:
        source = list(values)
        if not source:
            source = defaults

    return [source[index % len(source)] for index in range(count)]


def _ensure_parent_dir(save_path: PathLike) -> Path:
    path = Path(save_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def make_nature_rcparams(
    font_size: float = DEFAULT_FONT_SIZE_PT,
    use_grid: bool = False,
) -> dict[str, Any]:
    _validate_font_size(font_size)
    tick_size = max(5.0, font_size - 0.3)
    legend_size = max(5.0, font_size - 0.3)

    return {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": font_size,
        "axes.labelsize": font_size,
        "xtick.labelsize": tick_size,
        "ytick.labelsize": tick_size,
        "legend.fontsize": legend_size,
        "axes.linewidth": DEFAULT_AXES_LINEWIDTH,
        "xtick.major.width": DEFAULT_AXES_LINEWIDTH,
        "ytick.major.width": DEFAULT_AXES_LINEWIDTH,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": DEFAULT_LINE_WIDTH,
        "lines.markersize": DEFAULT_MARKER_SIZE,
        "grid.linewidth": DEFAULT_GRID_LINEWIDTH,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.grid": use_grid,
        "axes.axisbelow": True,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=NATURE_COLOR_CYCLE),
        "legend.frameon": False,
        "figure.dpi": DEFAULT_FIGURE_DPI,
        "savefig.dpi": DEFAULT_DPI,
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "hatch.linewidth": 0.6,
    }


@contextmanager
def nature_style_context(
    font_size: float = DEFAULT_FONT_SIZE_PT,
    use_grid: bool = False,
) -> Iterator[None]:
    with mpl.rc_context(make_nature_rcparams(font_size=font_size, use_grid=use_grid)):
        yield


def save_figure(fig: Figure, save_path: PathLike, dpi: int = DEFAULT_DPI) -> Path:
    output_path = _ensure_parent_dir(save_path)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)
    return output_path


def plot_line_chart(
    x_values: Sequence[float],
    y_values: Sequence[float] | Sequence[Sequence[float]],
    save_path: PathLike,
    labels: Sequence[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colors: Sequence[str] | None = None,
    markers: Sequence[str] | None = None,
    linestyles: Sequence[str] | None = None,
    figsize: FigureSize | Sequence[float] | None = "single",
    font_size: float = DEFAULT_FONT_SIZE_PT,
    use_grid: bool = True,
    legend: bool = True,
    x_lim: tuple[float, float] | None = None,
    y_lim: tuple[float, float] | None = None,
    x_ticks: Sequence[float] | None = None,
    y_ticks: Sequence[float] | None = None,
    y_errors: Sequence[float] | Sequence[Sequence[float]] | None = None,
    marker_size: float = DEFAULT_MARKER_SIZE,
    line_width: float = DEFAULT_LINE_WIDTH,
) -> Path:
    x_array = np.asarray(x_values, dtype=float)
    if x_array.ndim != 1 or x_array.size == 0:
        raise ValueError("x_values must be a non-empty 1D sequence.")

    series_list = _normalize_series_count(y_values, reference_length=x_array.size)
    if labels is not None and len(labels) != len(series_list):
        raise ValueError("labels length must match the number of line series.")

    if y_errors is None:
        error_list = [None] * len(series_list)
    else:
        raw_errors = _normalize_series_count(y_errors, reference_length=x_array.size)
        if len(raw_errors) != len(series_list):
            raise ValueError("y_errors must match the number of line series.")
        error_list = raw_errors

    color_list = _broadcast_style_values(colors, len(series_list), NATURE_COLOR_CYCLE)
    marker_list = _broadcast_style_values(markers, len(series_list), NATURE_MARKERS)
    linestyle_list = _broadcast_style_values(
        linestyles, len(series_list), NATURE_LINESTYLES
    )

    with nature_style_context(font_size=font_size, use_grid=use_grid):
        fig, ax = plt.subplots(figsize=resolve_figsize(figsize))

        for index, series in enumerate(series_list):
            if series.shape[0] != x_array.size:
                raise ValueError("Each y series must have the same length as x_values.")
            plot_label = labels[index] if labels is not None else None
            ax.errorbar(
                x_array,
                series,
                yerr=error_list[index],
                label=plot_label,
                color=color_list[index],
                marker=marker_list[index],
                linestyle=linestyle_list[index],
                linewidth=line_width,
                markersize=marker_size,
                markerfacecolor=color_list[index],
                markeredgecolor="white",
                markeredgewidth=0.5,
                capsize=2.5 if error_list[index] is not None else 0,
            )

        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if x_lim is not None:
            ax.set_xlim(x_lim)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if x_ticks is not None:
            ax.set_xticks(x_ticks)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        if legend and labels is not None:
            ax.legend()

        fig.tight_layout()
        return save_figure(fig, save_path)


def plot_bar_chart(
    values: Sequence[float] | Sequence[Sequence[float]],
    save_path: PathLike,
    x_labels: Sequence[str] | None = None,
    series_labels: Sequence[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colors: Sequence[str] | None = None,
    hatches: Sequence[str] | None = None,
    figsize: FigureSize | Sequence[float] | None = "single",
    font_size: float = DEFAULT_FONT_SIZE_PT,
    use_grid: bool = True,
    legend: bool = True,
    y_lim: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
    bar_width: float | None = None,
    use_default_hatches: bool = False,
    fill: bool = True,
    edgecolor: str | Sequence[str] | None = None,
) -> Path:
    values_array = np.asarray(values, dtype=float)
    if values_array.ndim == 1:
        values_array = values_array[np.newaxis, :]
    elif values_array.ndim != 2:
        raise ValueError("values must be a 1D or 2D numeric array.")

    if x_labels is not None:
        x_labels = list(x_labels)
        if (
            len(x_labels) == values_array.shape[0]
            and len(x_labels) != values_array.shape[1]
        ):
            values_array = values_array.T
        elif len(x_labels) != values_array.shape[1]:
            raise ValueError("x_labels length must match the number of categories.")

    n_series, n_categories = values_array.shape
    if series_labels is not None and len(series_labels) != n_series:
        raise ValueError("series_labels length must match the number of bar series.")

    if x_labels is None:
        x_labels = [str(index + 1) for index in range(n_categories)]

    color_list = _broadcast_style_values(colors, n_series, NATURE_COLOR_CYCLE)
    hatch_defaults = NATURE_HATCH_PATTERNS if use_default_hatches else ("",)
    hatch_list = _broadcast_style_values(hatches, n_series, hatch_defaults)

    width = bar_width if bar_width is not None else min(0.8 / max(n_series, 1), 0.32)
    centers = np.arange(n_categories, dtype=float)

    with nature_style_context(font_size=font_size, use_grid=use_grid):
        fig, ax = plt.subplots(figsize=resolve_figsize(figsize))

        for series_index in range(n_series):
            offset = (series_index - (n_series - 1) / 2.0) * width
            if edgecolor is None:
                bar_edgecolor = "black"
            elif isinstance(edgecolor, str):
                bar_edgecolor = edgecolor
            else:
                bar_edgecolor = edgecolor[series_index % len(edgecolor)]
            bars = ax.bar(
                centers + offset,
                values_array[series_index],
                width=width,
                color=color_list[series_index] if fill else "none",
                edgecolor=bar_edgecolor,
                linewidth=0.6,
                label=series_labels[series_index]
                if series_labels is not None
                else None,
            )
            hatch_pattern = hatch_list[series_index]
            if hatch_pattern:
                for bar in bars:
                    bar.set_hatch(hatch_pattern)

        ax.set_xticks(centers)
        ax.set_xticklabels(x_labels)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)
        if legend and series_labels is not None:
            ax.legend()

        fig.tight_layout()
        return save_figure(fig, save_path)


def plot_boxplot(
    values: Sequence[Sequence[float]],
    save_path: PathLike,
    labels: Sequence[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colors: Sequence[str] | None = None,
    hatches: Sequence[str] | None = None,
    figsize: FigureSize | Sequence[float] | None = "single",
    font_size: float = DEFAULT_FONT_SIZE_PT,
    use_grid: bool = True,
    y_lim: tuple[float, float] | None = None,
    y_ticks: Sequence[float] | None = None,
    showfliers: bool = False,
    use_default_hatches: bool = False,
) -> Path:
    box_data = _normalize_box_data(values)
    n_boxes = len(box_data)
    if labels is not None and len(labels) != n_boxes:
        raise ValueError("labels length must match the number of boxplot groups.")

    color_list = _broadcast_style_values(colors, n_boxes, NATURE_COLOR_CYCLE)
    hatch_defaults = NATURE_HATCH_PATTERNS if use_default_hatches else ("",)
    hatch_list = _broadcast_style_values(hatches, n_boxes, hatch_defaults)

    with nature_style_context(font_size=font_size, use_grid=use_grid):
        fig, ax = plt.subplots(figsize=resolve_figsize(figsize))

        artists = ax.boxplot(
            box_data,
            patch_artist=True,
            showfliers=showfliers,
            medianprops={"color": NATURE_PALETTE["black"], "linewidth": 1.0},
            whiskerprops={"linewidth": 0.8},
            capprops={"linewidth": 0.8},
            boxprops={"linewidth": 0.8},
        )

        for index, patch in enumerate(artists["boxes"]):
            patch.set_facecolor(color_list[index])
            patch.set_edgecolor("black")
            if hatch_list[index]:
                patch.set_hatch(hatch_list[index])

        if labels is not None:
            ax.set_xticks(np.arange(1, n_boxes + 1))
            ax.set_xticklabels(labels)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)
        if y_lim is not None:
            ax.set_ylim(y_lim)
        if y_ticks is not None:
            ax.set_yticks(y_ticks)

        fig.tight_layout()
        return save_figure(fig, save_path)


def _resolve_heatmap_cmap(
    cmap: str | Colormap | None,
    center: float | None,
) -> Colormap:
    if isinstance(cmap, Colormap):
        return cmap
    if cmap == "nature_diverging":
        return NATURE_DIVERGING_CMAP
    if cmap == "nature_sequential" or cmap is None:
        return NATURE_DIVERGING_CMAP if center is not None else NATURE_SEQUENTIAL_CMAP
    return mpl.colormaps[cmap]


def _build_heatmap_norm(
    data: np.ndarray,
    center: float | None,
    vmin: float | None,
    vmax: float | None,
) -> Normalize:
    data_min = float(np.nanmin(data)) if vmin is None else float(vmin)
    data_max = float(np.nanmax(data)) if vmax is None else float(vmax)

    if center is not None:
        center_value = float(center)
        data_min = min(data_min, center_value)
        data_max = max(data_max, center_value)
    if np.isclose(data_min, data_max):
        padding = 1.0 if data_min == 0 else abs(data_min) * 0.01
        data_min -= padding
        data_max += padding

    if center is not None:
        center_value = float(center)
        if not data_min < center_value:
            padding = max((data_max - data_min) * 0.05, 1e-6)
            data_min = center_value - padding
        if not center_value < data_max:
            padding = max((data_max - data_min) * 0.05, 1e-6)
            data_max = center_value + padding
        return TwoSlopeNorm(vcenter=center_value, vmin=data_min, vmax=data_max)
    return Normalize(vmin=data_min, vmax=data_max)


def _pick_annotation_text_color(cmap: Colormap, norm: Normalize, value: float) -> str:
    red, green, blue, _ = cmap(float(norm(value)))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.55 else "white"


def plot_heatmap(
    values: Sequence[Sequence[float]],
    save_path: PathLike,
    x_labels: Sequence[str] | None = None,
    y_labels: Sequence[str] | None = None,
    x_label: str | None = None,
    y_label: str | None = None,
    colorbar_label: str | None = None,
    cmap: str | Colormap | None = None,
    figsize: FigureSize | Sequence[float] | None = "heatmap",
    font_size: float = DEFAULT_FONT_SIZE_PT,
    annotate: bool = True,
    annotation_fmt: str = ".2f",
    center: float | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    data = np.asarray(values, dtype=float)
    if data.ndim != 2 or data.size == 0:
        raise ValueError("values must be a non-empty 2D numeric array.")

    if x_labels is not None and len(x_labels) != data.shape[1]:
        raise ValueError("x_labels length must match the number of columns.")
    if y_labels is not None and len(y_labels) != data.shape[0]:
        raise ValueError("y_labels length must match the number of rows.")

    with nature_style_context(font_size=font_size, use_grid=False):
        fig, ax = plt.subplots(figsize=resolve_figsize(figsize))
        selected_cmap = _resolve_heatmap_cmap(cmap=cmap, center=center)
        selected_norm = _build_heatmap_norm(
            data=data, center=center, vmin=vmin, vmax=vmax
        )

        image = ax.imshow(data, cmap=selected_cmap, norm=selected_norm, aspect="auto")

        if x_labels is not None:
            ax.set_xticks(np.arange(data.shape[1]))
            ax.set_xticklabels(x_labels)
        if y_labels is not None:
            ax.set_yticks(np.arange(data.shape[0]))
            ax.set_yticklabels(y_labels)
        if x_label is not None:
            ax.set_xlabel(x_label)
        if y_label is not None:
            ax.set_ylabel(y_label)

        if annotate:
            for row in range(data.shape[0]):
                for col in range(data.shape[1]):
                    value = data[row, col]
                    if np.isnan(value):
                        continue
                    ax.text(
                        col,
                        row,
                        format(value, annotation_fmt),
                        ha="center",
                        va="center",
                        color=_pick_annotation_text_color(
                            selected_cmap, selected_norm, float(value)
                        ),
                    )

        colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        if colorbar_label is not None:
            colorbar.set_label(colorbar_label)

        fig.tight_layout()
        return save_figure(fig, save_path)


__all__ = [
    "DEFAULT_DPI",
    "FIGURE_SIZE_PRESETS",
    "NATURE_COLOR_CYCLE",
    "NATURE_DIVERGING_CMAP",
    "NATURE_HATCH_PATTERNS",
    "NATURE_LINESTYLES",
    "NATURE_MARKERS",
    "NATURE_PALETTE",
    "NATURE_SEQUENTIAL_CMAP",
    "make_nature_rcparams",
    "mm_to_inch",
    "nature_style_context",
    "plot_bar_chart",
    "plot_boxplot",
    "plot_heatmap",
    "plot_line_chart",
    "resolve_figsize",
    "save_figure",
]
