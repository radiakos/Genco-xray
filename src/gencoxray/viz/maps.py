"""Static grid maps: the stress and accuracy map, and helpers shared with the mismatch animation.

Colours (``viz.style``): stress uses the single-hue sequential blue ramp (light = low); low trust is shown by line
style (dashed) and width, never by colour alone.
"""

from __future__ import annotations

import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from .style import ACCENT, INK, INK_2, INK_MUTED, SEQ_BLUE, SURFACE, mpl, seq_cmap


def draw_grid(ax, xy: np.ndarray, branch_from: np.ndarray, branch_to: np.ndarray, *,
              branch_values: np.ndarray | None = None, bus_values: np.ndarray | None = None,
              branch_dashed: np.ndarray | None = None, cmap=None, vmin=None, vmax=None,
              lw: float | np.ndarray = 0.8, bus_size: float = 4.0, outaged: np.ndarray | None = None,
              bus_cmap=None, bus_vmin=None, bus_vmax=None):
    """Draw branches as LineCollections (fast for ~3k lines) and buses as a scatter.

    branch_dashed: bool mask -> dashed linestyle (low trust). outaged: bool mask -> grey dotted.
    lw: scalar or per-branch widths. Returns (solid_line_collection, bus_scatter) for colourbars.
    No axis ticks; equal aspect.
    """
    from matplotlib.collections import LineCollection

    cmap = cmap or seq_cmap()
    nb = len(branch_from)
    segs = np.stack([xy[branch_from], xy[branch_to]], axis=1)
    lw = np.broadcast_to(np.asarray(lw, float), (nb,))
    dashed = np.zeros(nb, bool) if branch_dashed is None else np.asarray(branch_dashed, bool)
    out = np.zeros(nb, bool) if outaged is None else np.asarray(outaged, bool)
    vals = None if branch_values is None else np.asarray(branch_values, float)
    if vals is not None:
        vmin = np.nanmin(vals) if vmin is None else vmin
        vmax = np.nanmax(vals) if vmax is None else vmax
    solid_lc = None
    # draw low values first so stressed branches sit on top
    order = np.argsort(np.nan_to_num(vals, nan=-np.inf)) if vals is not None else np.arange(nb)
    for style, sel in (("solid", ~dashed & ~out), ((0, (2.2, 1.6)), dashed & ~out)):
        idx = order[sel[order]]
        if idx.size == 0:
            continue
        lc = LineCollection(segs[idx], linewidths=lw[idx], linestyles=style, capstyle="round", zorder=2)
        if vals is not None:
            lc.set_array(vals[idx])
            lc.set_cmap(cmap)
            lc.set_clim(vmin, vmax)
        else:
            lc.set_color(INK_MUTED)
        ax.add_collection(lc)
        if style == "solid":
            solid_lc = lc
    if out.any():
        ax.add_collection(LineCollection(segs[out], linewidths=0.9, linestyles=(0, (1, 1.5)), colors=INK_MUTED,
                                         zorder=1))
    sc = None
    if bus_values is not None:
        sc = ax.scatter(xy[:, 0], xy[:, 1], c=bus_values, s=bus_size, cmap=bus_cmap or cmap, vmin=bus_vmin,
                        vmax=bus_vmax, linewidths=0, zorder=3)
    elif bus_size > 0:
        ax.scatter(xy[:, 0], xy[:, 1], s=bus_size, color=INK_MUTED, linewidths=0, zorder=3, alpha=0.6)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    pad = 0.02
    ax.set_xlim(xy[:, 0].min() - pad, xy[:, 0].max() + pad)
    ax.set_ylim(xy[:, 1].min() - pad, xy[:, 1].max() + pad)
    return solid_lc, sc


def joint_map_figure(joint_df: pd.DataFrame, xy: np.ndarray, out_path: str, title: str = "", *,
                     stress_label: str = "branch stress (GENCO)", stress_q: float = 0.9, trust_q: float = 1 / 3,
                     note: str = "") -> list:
    """Stress and accuracy map: branches coloured by stress_genco (sequential), dashed where quadrant is *_untrusted, thick for
    stressed_untrusted. Inset: stress vs trust scatter with quadrant lines. Save PNG (dpi 200) + PDF."""
    plt = mpl()
    from matplotlib.lines import Line2D

    jt = joint_df.sort_index()
    q = jt.quadrant.astype(str).to_numpy()
    untrusted = np.char.endswith(q.astype(str), "_untrusted")
    hot = q == "stressed_untrusted"
    stressed = np.char.startswith(q.astype(str), "stressed")
    s = jt.stress_genco.to_numpy(float)
    lw = np.where(hot, 2.4, np.where(stressed, 1.5, 0.55))

    fig = plt.figure(figsize=(13.5, 9.0))
    fig.patch.set_facecolor(SURFACE)
    fig.text(0.015, 0.975, title, fontsize=11.5, color=INK, va="top", ha="left")
    ax = fig.add_axes((0.01, 0.05, 0.68, 0.84))
    ax.set_facecolor(SURFACE)
    lc, _ = draw_grid(ax, xy, jt.from_bus.to_numpy(), jt.to_bus.to_numpy(), branch_values=s, branch_dashed=untrusted,
                      lw=lw, bus_size=1.2, vmin=np.nanquantile(s, 0.02), vmax=np.nanquantile(s, 0.995))
    cax = fig.add_axes((0.735, 0.30, 0.235, 0.016))
    cb = fig.colorbar(lc, cax=cax, orientation="horizontal")
    cb.outline.set_visible(False)
    cb.ax.tick_params(labelsize=7.5, colors=INK_2)
    cb.set_label(stress_label, fontsize=8, color=INK_2)

    # side panel: stress vs trust with quadrant lines (trust axis clipped to its 2nd-100th percentile)
    ia = fig.add_axes((0.745, 0.50, 0.225, 0.34))
    ia.set_facecolor(SURFACE)
    for sp in ("top", "right"):
        ia.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ia.spines[sp].set_color(INK_MUTED)
    t = jt.trust.to_numpy(float)
    ok = np.isfinite(s) & np.isfinite(t)
    ia.scatter(s[ok & ~hot], t[ok & ~hot], s=3, color=INK_MUTED, alpha=0.35, linewidths=0)
    ia.scatter(s[ok & hot], t[ok & hot], s=9, color=ACCENT, linewidths=0, zorder=3)
    s_cut, t_cut = np.nanquantile(s[ok], stress_q), np.nanquantile(t[ok], trust_q)
    ia.axvline(s_cut, color=INK_2, lw=0.8, ls=(0, (3, 2)))
    ia.axhline(t_cut, color=INK_2, lw=0.8, ls=(0, (3, 2)))
    t_lo = np.nanquantile(t[ok], 0.02)
    t_hi = np.nanmax(t[ok])
    ia.set_ylim(t_lo - 0.05 * (t_hi - t_lo), t_hi + 0.05 * (t_hi - t_lo))
    n_clip = int((t[ok] < t_lo).sum())
    ia.text(0.98, 0.10, f"stressed and error-prone: n = {int(hot.sum())}", transform=ia.transAxes, ha="right",
            va="bottom", fontsize=7.5, color=INK)
    ia.set_title("tail loading against accuracy score (cut-offs dashed)", fontsize=8.5, color=INK, loc="left")
    ia.set_xlabel(stress_label, fontsize=7.5, color=INK_2)
    ia.set_ylabel("accuracy score (−mean error score)", fontsize=7.5, color=INK_2)
    ia.tick_params(labelsize=7, colors=INK_2)
    if n_clip:
        ia.text(0.02, 0.03, f"{n_clip} branches below the axis", transform=ia.transAxes, fontsize=6.5,
                color=INK_MUTED)

    handles = [Line2D([], [], color=SEQ_BLUE[9], lw=1.5), Line2D([], [], color=SEQ_BLUE[9], lw=1.5, ls=(0, (2.2, 1.6))),
               Line2D([], [], color=SEQ_BLUE[11], lw=2.6, ls=(0, (2.2, 1.6)))]
    fig.legend(handles, ["accurate", f"error-prone (bottom {trust_q:.0%} of accuracy score)",
                         f"stressed (top {1 - stress_q:.0%} of tail loading) and error-prone"],
               loc="upper left", bbox_to_anchor=(0.735, 0.24), ncol=1, frameon=False, fontsize=8.5, labelcolor=INK_2)
    if note:
        fig.text(0.735, 0.06, "\n".join(textwrap.wrap(note, 62)), fontsize=7, color=INK_MUTED, va="bottom")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    paths = [out.with_suffix(".png"), out.with_suffix(".pdf")]
    for p in paths:
        fig.savefig(p, dpi=200, facecolor=SURFACE)
    plt.close(fig)
    return paths
