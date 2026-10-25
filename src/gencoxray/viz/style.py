"""Shared figure style: the dataviz reference palette and matplotlib helpers.

At most three categorical slots per chart (blue, orange, aqua: the slots that pass all-pairs colour-vision checks);
stress uses the single-hue sequential blue ramp (light = low). Identity never relies on colour alone.
"""

from __future__ import annotations

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#8a8985"
GRID = "#e6e5e1"
SERIES = ("#2a78d6", "#eb6834", "#1baf7a")  # categorical slots 1-3, fixed order
ACCENT = SERIES[1]  # orange, used once for the highlighted quadrant in the stress and accuracy map inset
SEQ_BLUE = ("#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf",
            "#1c5cab", "#184f95", "#104281", "#0d366b")


def mpl():
    """pyplot with the non-interactive Agg backend."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def seq_cmap(name: str = "seq_blue"):
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(name, SEQ_BLUE)


def style_axes(ax) -> None:
    """Light surface, no top/right spines, muted ticks and grid."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK_MUTED)
    ax.tick_params(colors=INK_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
