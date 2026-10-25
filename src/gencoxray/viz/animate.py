"""Animation of the bus mismatch and branch loading over the correction steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import schemas
from ..features import residual_magnitude
from .maps import draw_grid
from .style import INK, INK_2, INK_MUTED, SURFACE, mpl, seq_cmap


def residual_animation(traces: dict[str, np.ndarray], scenario: int, xy: np.ndarray, out_path: str, *,
                       fps: int = 2, vmin_log: float = -3.0, vmax_log: float = 2.0,
                       struct_p: np.ndarray | None = None, struct_q: np.ndarray | None = None,
                       title: str = "") -> list[Path]:
    """T frames. Bus colour = log10 ||PBRes_t|| (structural components masked) on a FIXED colour scale across
    frames; branches coloured by loading_t; outaged branches grey and dotted. The title shows the step t/T and the
    mean residual. Write with matplotlib.animation.PillowWriter (GIF). Also save the last frame as PNG.

    struct_p / struct_q: the scenario's own structural masks [B] (bus types vary per scenario); default: from
    traces['bus_type'].
    """
    plt = mpl()
    from matplotlib.animation import FuncAnimation, PillowWriter
    from matplotlib.colors import LinearSegmentedColormap

    i = int(np.flatnonzero(np.asarray(traces["scenario"]) == scenario)[0])
    if struct_p is None or struct_q is None:
        struct_p, struct_q = schemas.structural_zero_mask(np.asarray(traces["bus_type"]).astype(str))
    r = residual_magnitude(traces["res_p"][i], traces["res_q"][i], struct_p, struct_q)  # [T, B] MVA
    T = r.shape[0]
    logr = np.log10(np.clip(r, 10**vmin_log, None))
    E = traces["loading"].shape[2] // 2
    ld = traces["loading"][i]  # [T, 2E]
    L = np.fmax(ld[:, :E], ld[:, E:])  # [T, E] physical branch loading
    out = ~np.isfinite(L).any(0)
    ef, et = np.asarray(traces["edge_from"])[:E], np.asarray(traces["edge_to"])[:E]
    warm = LinearSegmentedColormap.from_list("res", ["#f0efec", "#f5b38f", "#eb6834", "#b23b0f", "#6b1f05"])

    fig = plt.figure(figsize=(8.6, 8.0))
    fig.patch.set_facecolor(SURFACE)
    ax = fig.add_axes((0.02, 0.09, 0.96, 0.84))

    def frame(t: int):
        ax.clear()
        ax.set_facecolor(SURFACE)
        lc, sc = draw_grid(ax, xy, ef, et, branch_values=L[t], outaged=out, lw=0.6, cmap=seq_cmap(),
                           vmin=0.0, vmax=1.0, bus_values=logr[t], bus_size=7, bus_cmap=warm,
                           bus_vmin=vmin_log, bus_vmax=vmax_log)
        mean_r = float(np.nanmean(r[t]))
        ax.set_title(f"{title}   correction step {t + 1}/{T}   mean mismatch = {mean_r:.3g} MVA", fontsize=10,
                     color=INK, loc="left")
        return lc, sc

    lc, sc = frame(0)
    cax1 = fig.add_axes((0.06, 0.05, 0.38, 0.016))
    cb1 = fig.colorbar(sc, cax=cax1, orientation="horizontal")
    cb1.set_label("bus mismatch, log₁₀ of |(ΔP, ΔQ)| [MVA] (fixed scale)", fontsize=8, color=INK_2)
    cax2 = fig.add_axes((0.56, 0.05, 0.38, 0.016))
    cb2 = fig.colorbar(lc, cax=cax2, orientation="horizontal")
    cb2.set_label("branch loading at this step (grey dotted = outaged)", fontsize=8, color=INK_2)
    for cb in (cb1, cb2):
        cb.outline.set_visible(False)
        cb.ax.tick_params(labelsize=7.5, colors=INK_2)
    fig.text(0.02, 0.985, "The twelve intermediate solutions of GENCO (recording hooks); colour scales fixed across frames.",
             fontsize=6.8, color=INK_MUTED)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    anim = FuncAnimation(fig, frame, frames=T, blit=False)
    gif = out_path.with_suffix(".gif")
    anim.save(gif, writer=PillowWriter(fps=fps), dpi=90, savefig_kwargs={"facecolor": SURFACE})
    frame(T - 1)
    png = out_path.with_name(out_path.stem + "_last_step.png")
    fig.savefig(png, dpi=160, facecolor=SURFACE)
    plt.close(fig)
    return [gif, png]
