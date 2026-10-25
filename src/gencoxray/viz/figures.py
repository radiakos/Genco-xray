"""Submission figures from results/ only. matplotlib, static PNG + PDF, light surface. Earlier figure ids remain
accepted as aliases (:data:`ALIASES`).

Colours (``viz.style``): at most three categorical slots per chart; remaining series are grey and distinguished by
line style and direct labels, so identity never relies on colour alone. Every figure function has the signature
``f(grid, out_dir=None) -> list[Path]`` and is registered in :data:`FIGURES`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .. import io
from .style import ACCENT, GRID, INK, INK_2, INK_MUTED, SEQ_BLUE, SERIES, SURFACE, mpl
from .style import style_axes as _style

NEWTON_COLOUR = "#8a5cd0"  # one extra categorical colour, used only for the Newton-step rule
# gate -> (label, colour, linestyle, linewidth, z); gates missing from e4_curves.csv are skipped
F5_GATES = {
    "newton_margin": ("one Newton step, then distance to the limit", NEWTON_COLOUR, "-", 2.2, 6),
    "margin_trust": ("margin in units of expected error", SERIES[0], "-", 2.2, 5),
    "last_step_margin": ("margin in units of last-step change", SERIES[0], (0, (3, 2)), 1.6, 5),
    "margin": ("distance to the limit only", SERIES[1], "-", 2.0, 4),
    "trust": ("scenario accuracy score", SERIES[2], "-", 2.0, 4),
    "residual": ("mean final mismatch", INK_MUTED, "-", 1.4, 3),
    "k": ("number of outages", INK_MUTED, ":", 1.4, 3),
    "random": ("random", INK_MUTED, (0, (1, 3)), 1.2, 2),
    "oracle": ("oracle", INK, "--", 1.4, 3),
}


def _out_dir(out_dir: str | Path | None) -> Path:
    return Path(out_dir) if out_dir else io.results_dir("figures")


def _save(fig, out_dir, grid: str, name: str) -> list[Path]:
    """Save ``<name>_<grid>.{png,pdf}`` (dpi 200, tight bounding box) and close the figure."""
    out = _out_dir(out_dir)
    paths = [out / f"{name}_{grid}.{ext}" for ext in ("png", "pdf")]
    for p in paths:
        fig.savefig(p, dpi=200, facecolor=SURFACE, bbox_inches="tight")
    mpl().close(fig)
    return paths


_GRID_NAMES = {"texas2k": "Texas 2000-bus", "case118": "IEEE 118-bus", "case118_small": "IEEE 118-bus, small checkpoint",
               "case118_seed1": "IEEE 118-bus, seed 1", "case118_seed42": "IEEE 118-bus, seed 42"}
_WORDS = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "eleven", "twelve",
          "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty")


def _grid_name(grid: str) -> str:
    return _GRID_NAMES.get(grid, grid)


def _range_label(sub: str, grid: str) -> str:
    """Readable outage-count range of a hybrid / stress subset name (all, in_dist, k_le_5, k_ge_10, k3-5)."""
    import re

    from ..config import k_fit

    if sub == "all":
        return "all outage counts"
    if sub == "in_dist":
        k = k_fit(grid)
        return f"up to {k} outage{'s' if k > 1 else ''}"
    for pat, fmt in ((r"k_le_(\d+)", "up to {} outages"), (r"k_ge_(\d+)", "{} or more outages")):
        m = re.fullmatch(pat, sub)
        if m:
            return fmt.format(m[1])
    m = re.fullmatch(r"k(\d+)-(\d+)", sub)
    return f"{int(m[1])}–{int(m[2])} outages" if m else sub


def f5_hybrid(grid: str = "texas2k", subsets: tuple[str, ...] = ("k_le_5", "k_ge_10"),
              out_dir: str | Path | None = None) -> list[Path]:
    """Selecting cases for AC re-solve: overload recall and max |loading error| vs share re-solved with AC, one
    column per subset. No timing: the saving is the share of AC solves, which does not depend on hardware."""
    plt = mpl()
    curves = pd.read_csv(io.results_dir(grid) / "e4_curves.csv")
    summ = io.read_summary()["e4"][grid]
    target = summ["recall_target"]
    fig, axes = plt.subplots(2, len(subsets), figsize=(4.6 * len(subsets) + 0.9, 6.2), sharex=True,
                             gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    axes = np.atleast_2d(axes).reshape(2, len(subsets))
    for j, sub in enumerate(subsets):
        c = curves[curves.subset == sub]
        ss = summ["subsets"][sub]
        ax_r, ax_e = axes[0, j], axes[1, j]
        for ax in (ax_r, ax_e):
            _style(ax)
        r_lo = max(0.0, np.floor(min(ss["genco_only"]["recall"], 0.9) * 20) / 20)
        for gate, (label, col, ls, lw, z) in F5_GATES.items():
            d = c[c.gate == gate].sort_values("q")
            if d.empty:
                continue
            ax_r.plot(d.q, d.recall, color=col, ls=ls, lw=lw, zorder=z, solid_capstyle="round")
            e = d.max_abs_err.clip(lower=1e-3)
            ax_e.plot(d.q, e, color=col, ls=ls, lw=lw, zorder=z, solid_capstyle="round")
            qs = (ss["gates"].get(gate) or ss.get("selection_rules", {}).get(gate, {})).get("q_star")
            if gate in ("newton_margin", "margin_trust", "margin", "trust", "oracle") and qs is not None:
                ax_r.plot([qs], [target], marker="o", ms=6.5, color=col, mec=SURFACE, mew=1.5, zorder=z + 1)
                if gate == "newton_margin":  # q* = 0: the dot on the y axis says it; a label would hit the oracle's
                    continue
                ax_r.annotate(f"{qs:.2f}", (qs, target), xytext=(0, -11), textcoords="offset points", ha="center",
                              va="top", fontsize=7.5, color=INK, zorder=7,
                              bbox={"boxstyle": "round,pad=0.15", "fc": SURFACE, "ec": "none", "alpha": 0.85})
        ax_r.axhline(target, color=INK_2, lw=0.9, ls=(0, (4, 3)), zorder=1)
        ax_r.set_ylim(r_lo, 1.004)
        ax_e.set_yscale("log")
        ax_e.set_xlim(0, 1)
        ax_r.set_title(f"{_grid_name(grid)}, {_range_label(sub, grid)}  ({ss['n_scenarios']:,} scenarios, "
                       f"{ss['n_overload_true']:,} overloaded)", fontsize=9, color=INK, loc="left", pad=8)
        ax_e.set_xlabel("share of scenarios re-solved with AC  (q)", fontsize=8.5, color=INK_2)
        if j == 0:
            ax_r.set_ylabel("recall of overloaded scenarios", fontsize=8.5, color=INK_2)
            ax_e.set_ylabel("largest loading error among cases kept", fontsize=8.5, color=INK_2)
    shown = {g: v for g, v in F5_GATES.items() if g in set(curves.gate)}
    handles = [plt.Line2D([], [], color=col, ls=ls, lw=lw) for (_, col, ls, lw, _) in shown.values()]
    handles.append(plt.Line2D([], [], color=INK_2, ls=(0, (4, 3)), lw=0.9))
    labels = [v[0] for v in shown.values()] + [f"{100 * target:.1f} % recall target"]
    fig.legend(handles, labels, loc="lower center", ncol=min(len(labels), 5), frameon=False,
               fontsize=8, labelcolor=INK_2, bbox_to_anchor=(0.5, -0.005))
    q10 = summ["subsets"]["k_ge_10"]["gates"]["margin_trust"]["q_star"]
    note = ("Dots and numbers: q*, the fraction re-solved with AC to reach the target. The saving in AC solves is at "
            f"most 1/q* (with 10 or more outages: q* = {q10:.2f}, at most {1 / q10:.1f}).\n"
            "The Newton-step rule decides "
            "the cases it keeps from the corrected loadings and costs one Jacobian factorization per scenario.")
    fig.text(0.01, 0.075, note, fontsize=6.8, color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    return _save(fig, out_dir, grid, "ac_resolve_selection")


def f6_stress_agreement(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Stress agreement: stress maps from GENCO and DC vs AC. Top: per-branch CVaR95 of loading, GENCO vs AC and DC vs AC (all
    scenarios, identical axes, identity line). Bottom: Spearman and precision@25 of the CVaR95 ranking vs AC per
    k-bin, GENCO vs DC, with 95 % cluster-bootstrap CIs."""
    plt = mpl()
    base = io.results_dir(grid)
    st = pd.read_csv(base / "e5_branch_stress.csv")
    agr = pd.read_csv(base / "e5_agreement.csv")
    summ = io.read_summary()["e5"][grid]
    st = st[st.subset == "all"].pivot_table(index="branch", columns="source", values="cvar")
    src_style = {"genco": ("GENCO", SERIES[0], "o"), "dc": ("DC power flow", SERIES[1], "s")}

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 8.2), gridspec_kw={"height_ratios": [1.15, 1]})
    fig.patch.set_facecolor(SURFACE)
    lo = float(np.nanmin(st.values)) if st.size else 0.0
    hi = float(np.nanmax(st.values)) if st.size else 1.0
    pad = 0.03 * (hi - lo)
    lim = (lo - pad, hi + pad)
    a_all = agr[(agr.subset == "all")].set_index(["source", "metric", "stat"])
    for j, src in enumerate(("genco", "dc")):
        ax = axes[0, j]
        _style(ax)
        name, col, _ = src_style[src]
        if src in st:
            ax.plot(lim, lim, color=INK_MUTED, lw=0.9, ls=(0, (4, 3)), zorder=1)
            ax.axhline(1.0, color=GRID, lw=0.8, zorder=0)
            ax.axvline(1.0, color=GRID, lw=0.8, zorder=0)
            ax.scatter(st["ac"], st[src], s=9, color=col, alpha=0.55, linewidths=0, zorder=3)
            rho = a_all.loc[(src, "cvar", "spearman"), "point"]
            ax.text(0.03, 0.97, f"rank correlation {rho:.4f}  ({len(st):,} branches)", transform=ax.transAxes,
                    ha="left", va="top", fontsize=8, color=INK)
        ax.set_xlim(lim)
        ax.set_ylim(lim)
        ax.set_aspect("equal")
        ax.set_xlabel("AC: branch tail loading", fontsize=8.5, color=INK_2)
        ax.set_ylabel(f"{name}: branch tail loading", fontsize=8.5, color=INK_2)
        ax.set_title(f"{name} vs AC", fontsize=9.5, color=INK, loc="left")

    subs = [s for s in summ["subsets"]]
    x = np.arange(len(subs))
    for j, (stat, ylab) in enumerate((("spearman", "rank correlation with AC: GENCO − DC"),
                                      ("precision@25", "top-25 precision against AC: GENCO − DC"))):
        ax = axes[1, j]
        _style(ax)
        ax.axhline(0, color=INK_2, lw=0.9, zorder=1)
        d = pd.DataFrame({s: summ["subsets"][s]["h3"].get("paired_genco_minus_dc", {}).get(stat, {}) for s in subs}).T
        if not d.empty:
            ax.errorbar(x, d["delta"], yerr=[(d["delta"] - d["lo"]).clip(lower=0), (d["hi"] - d["delta"]).clip(lower=0)],
                        fmt="o", ms=6.5, color=SERIES[0], mec=SURFACE, mew=1.2, elinewidth=1.6, capsize=0, zorder=3)
            for xi, (dv, lo_) in enumerate(zip(d["delta"], d["lo"])):
                ax.annotate(f"{dv:+.3f}", (xi, dv), xytext=(8, 0), textcoords="offset points", va="center",
                            fontsize=7.5, color=INK)
            span = max(float(np.nanmax(np.abs(d[["lo", "hi", "delta"]].to_numpy()))), 1e-3)
            ax.set_ylim(-1.25 * span, 1.25 * span)
        ax.set_xticks(x, [_range_label(s, grid).replace(" outages", "") for s in subs], fontsize=8)
        ax.set_ylabel(ylab, fontsize=8.5, color=INK_2)
        ax.set_xlim(-0.5, len(subs) - 0.5)
        ax.set_xlabel("number of outages", fontsize=8.5, color=INK_2)
        ax.text(0.99, 0.97, "GENCO closer to AC ↑", transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                color=INK_MUTED)
        if j == 0:
            ax.set_title("Branch ranking by tail loading: GENCO minus DC", fontsize=9.5, color=INK, loc="left")
    fig.text(0.01, 0.015, f"{_grid_name(grid)}: {summ['n_scenarios']:,} scenarios. Tail loading = mean of the worst 5 % of loadings. "
             "Top: one dot per branch, dashed line = identity. Bottom: 95 % intervals, resampling by load scenario.", fontsize=6.8,
             color=INK_MUTED)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    return _save(fig, out_dir, grid, "stress_agreement")


def f1_joint_map(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Stress and accuracy map: branches coloured by GENCO stress (CVaR95), dashed where error-prone, thick where
    stressed and error-prone; inset: stress vs accuracy score with the quadrant cut-offs."""
    from .maps import joint_map_figure

    jt = pd.read_parquet(io.results_dir(grid) / "joint.parquet").set_index("branch")
    sb = io.read_table("static_bus", grid)
    xy = sb[["x", "y"]].to_numpy(np.float64)
    e6 = io.read_summary()["e6"][grid]
    h = e6["cvar"]["h4"]["abs"]
    ratio = h["median_untrusted"] / h["median_trusted"] if h["median_trusted"] else float("nan")
    title = (f"{_grid_name(grid)}: branch stress from GENCO, and where GENCO is least accurate\n"
             f"Stressed, error-prone branches differ from AC {ratio:.1f} times more (ratio of medians; "
             f"p = {h['p_value']:.0e})")
    p = e6["protocol"]
    out = _out_dir(out_dir) / f"stress_and_accuracy_map_{grid}"
    note = (f"Stress = tail loading (mean of the worst 5 % of loadings) of GENCO over all scenarios; accuracy = "
            f"-mean error score over scenarios where predicted loading > {p['stressed_loading_for_trust']}. Layout: "
            "graph embedding (the data have no coordinates).")
    return joint_map_figure(jt, xy, str(out), title, stress_label="branch tail loading (GENCO)",
                            stress_q=p["stress_quantile"], trust_q=p["trust_quantile"], note=note)


def f7_disagreement(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Disagreement |stress_GENCO - stress_AC| (CVaR95) per stress × accuracy group, log scale, violin + median."""
    plt = mpl()
    jt = pd.read_parquet(io.results_dir(grid) / "joint.parquet")
    e6 = io.read_summary()["e6"][grid]
    order = ["stressed_untrusted", "stressed_trusted", "calm_untrusted", "calm_trusted"]
    labels = ["stressed,\nerror-prone", "stressed,\naccurate", "calm,\nerror-prone", "calm,\naccurate"]
    data = [np.log10(jt.loc[jt.quadrant == q, "disagreement"].clip(lower=1e-6).to_numpy()) for q in order]
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ok = [i for i, d in enumerate(data) if d.size >= 2]
    if ok:
        vp = ax.violinplot([data[i] for i in ok], positions=ok, widths=0.75, showextrema=False)
        for j, body in zip(ok, vp["bodies"]):
            body.set_facecolor(ACCENT if j == 0 else SERIES[0])
            body.set_alpha(0.55 if j < 2 else 0.25)
            body.set_edgecolor("none")
    lo = min(np.min(d) for d in data if d.size)
    hi = max(np.max(d) for d in data if d.size)
    for i, d in enumerate(data):
        if d.size:
            med = np.median(d)
            ax.plot([i - 0.22, i + 0.22], [med, med], color=INK, lw=1.6, solid_capstyle="round", zorder=3)
            ax.text(i + 0.26, med, f"{10 ** med:.1e}", va="center", fontsize=7.5, color=INK)
        ax.text(i, lo - 0.35, f"n = {d.size}", ha="center", va="top", fontsize=7, color=INK_2)
    ax.set_xticks(range(4), labels, fontsize=8.5)
    ax.set_ylabel("log$_{10}$ |tail loading, GENCO − AC|", fontsize=8.5, color=INK_2)
    h = e6["cvar"]["h4"]["abs"]
    ax.set_title(f"{_grid_name(grid)}: disagreement with AC by group\nstressed and error-prone vs stressed and accurate: one-sided Mann–Whitney "
                 f"p = {h['p_value']:.1g}, rank-biserial {h['rank_biserial']:.2f}", fontsize=9, color=INK, loc="left")
    rob = e6["cvar"].get("robustness", {}).get("ols_log10_disagreement", {})
    if rob:
        h4 = h["median_untrusted"] / h["median_trusted"]
        fig.text(0.01, -0.03, f"Bars: medians; ratio of medians (stressed, error-prone / stressed, accurate) = {h4:.1f}. "
                 f"Regression on stress level inside the stressed set: factor {rob['factor']:.2f} for error-prone "
                 "branches (a different quantity).", fontsize=7, color=INK_MUTED)
    ax.set_ylim(lo - 0.8, hi + 0.3)
    return _save(fig, out_dir, grid, "stress_disagreement_by_group")


def f2_anatomy(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Mismatch by step: median bus mismatch r_t vs correction step (IQR band) by bus type, hops to the outage and k."""
    plt = mpl()
    d = pd.read_csv(io.results_dir(grid) / "e1_anatomy.csv")
    d["group"] = d.group.astype(str)
    panels = [("bus_type", "bus type", None), ("hop_to_outage", "hops to the outaged element", ["0", "1", "2", "3-4", "5+"]),
              ("k", "number of outages", None)]
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, (g, title, order) in zip(axes, panels):
        _style(ax)
        sub = d[d.grouping == g]
        present = set(sub.group)
        groups = [x for x in (order or sorted(present, key=lambda v: (len(v), v))) if x in present]
        if g == "bus_type":
            cols = {grp: SERIES[i % 3] for i, grp in enumerate(groups)}
        else:  # ordered magnitude -> one-hue sequential ramp
            steps = np.linspace(3, len(SEQ_BLUE) - 1, max(len(groups), 1)).astype(int)
            cols = {grp: SEQ_BLUE[s] for grp, s in zip(groups, steps)}
        ends = []
        for grp in groups:
            c = sub[sub.group == grp].sort_values("step")
            ax.fill_between(c.step, c.q25, c.q75, color=cols[grp], alpha=0.13, linewidth=0)
            ax.plot(c.step, c["median"], color=cols[grp], lw=2, solid_capstyle="round")
            ends.append((np.log10(c["median"].iloc[-1]), grp, c.step.iloc[-1]))
        # end labels, nudged apart in log space so they never overlap
        ends.sort()
        ys = [e[0] for e in ends]
        for i in range(1, len(ys)):
            ys[i] = max(ys[i], ys[i - 1] + 0.15)
        for (y0, grp, x0), y in zip(ends, ys):
            ax.annotate(grp, (x0, 10 ** y0), xytext=(x0 + 0.25, 10 ** y), textcoords="data", va="center",
                        fontsize=7.5, color=INK)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlabel("correction step", fontsize=8.5, color=INK_2)
        ax.set_xticks(range(1, 13))
        ax.set_xlim(0.7, 13.4)
    axes[0].set_ylabel("mismatch per bus [MVA] (median, interquartile range)", fontsize=8.5, color=INK_2)
    fig.text(0.01, -0.03, f"{_grid_name(grid)}: 256 scenarios per outage count, for 1, 2, 5, 10 and 20 outages; components fixed by "
             "bus type are excluded.", fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, grid, "mismatch_by_step")


def f3_auroc(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Error-check comparison: AUROC [95 % CI] per check, inside vs beyond the training range, for the headline bus
    and branch labels."""
    plt = mpl()
    e2 = io.read_summary()["e2"][grid]
    rows = [("combiner", "error predictor"), ("combiner_traj_conf", "error predictor + position"),
            ("combiner_hgb", "error predictor (boosted trees)"), ("conf_only", "position only"),
            ("f_final", "final mismatch"), ("hop_to_outage", "hops to outage"), ("k", "number of outages"),
            ("dc_disagreement", "|GENCO − DC| loading"), ("random", "random")]
    best_names = {"nbmax_f_drift_vm": "last-step voltage change (neighbours)",
                  "f_drift_vm": "last-step voltage change (own bus)", "g_drift_tail_std": "last-step loading change"}
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8))
    fig.patch.set_facecolor(SURFACE)
    for ax, level in zip(axes, ("bus", "branch")):
        _style(ax)
        L = e2[level]
        tau = L["headline_tau"]
        m = L[f"tau_{tau}"]["metrics"]
        best = L[f"tau_{tau}"]["best_single_feature_in_dist"]
        rr = [r for r in rows if r[0] in m.get("in_dist", {})]
        if best:
            rr.append((best["score"], f"best single: {best_names.get(best['score'], best['score'])}"))
        y = np.arange(len(rr))[::-1]
        for off, (setname, col, mk, lab) in zip((0.16, -0.16), (("in_dist", SERIES[0], "o", "outage counts inside the training range"),
                                                               ("shift_pooled", SERIES[1], "s", "beyond the training range"))):
            for yi, (key, _) in zip(y, rr):
                v = m.get(setname, {}).get(key)
                if not v or v["auroc"] is None:
                    continue
                if v["auroc_lo"] is not None:
                    ax.plot([v["auroc_lo"], v["auroc_hi"]], [yi + off] * 2, color=col, lw=1.6, solid_capstyle="round")
                ax.plot([v["auroc"]], [yi + off], marker=mk, ms=5.5, color=col, mec=SURFACE, mew=1, ls="none",
                        label=lab if key == "combiner" else None)
        ax.axvline(0.5, color=INK_MUTED, lw=0.8, ls=(0, (3, 2)))
        ax.set_yticks(y, [r[1] for r in rr], fontsize=8)
        ax.set_xlim(0.2, 1.0)
        ax.set_xlabel("AUROC [95 % interval, resampling by load scenario]", fontsize=8.5, color=INK_2)
        thr = f"|V error| > {tau} p.u." if level == "bus" else f"|loading error| > {tau}"
        ax.set_title(f"{'bus voltage' if level == 'bus' else 'branch loading'} error: {thr}", fontsize=9.5, color=INK, loc="left")
    axes[0].legend(loc="upper left", frameon=False, fontsize=8, labelcolor=INK_2)
    fig.text(0.01, -0.03, f"{_grid_name(grid)}: fitted inside the training range only; each row is scored by a model that never "
             "saw its load scenario.", fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, grid, "error_check_comparison")


def f4_risk_coverage(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Error rate of kept rows vs share kept (bus, branch; inside the training range) and hit rate of the
    branch-loading error bands vs k."""
    plt = mpl()
    rc = pd.read_csv(io.results_dir(grid) / "e2_risk_coverage.csv")
    e3 = pd.read_csv(io.results_dir(grid) / "e3_conformal.csv")
    alpha = io.read_summary()["e3"][grid]["alpha"]
    fig, axes = plt.subplots(1, 3, figsize=(13.0, 3.9))
    fig.patch.set_facecolor(SURFACE)
    style = {"combiner": ("error predictor", SERIES[0], "-"), "conf_only": ("position only", SERIES[1], "-"),
             "f_final": ("final mismatch", SERIES[2], "-"), "random": ("random", INK_MUTED, (0, (1, 2)))}
    for ax, level in zip(axes[:2], ("bus", "branch")):
        _style(ax)
        for sc, (lab, col, ls) in style.items():
            c = rc[(rc.level == level) & (rc.score == sc)]
            ax.plot(c.coverage, c.risk, color=col, ls=ls, lw=1.8, label=lab)
        ax.set_xlabel("share of rows kept (lowest error score first)", fontsize=8.5, color=INK_2)
        ax.set_ylabel("error rate among kept rows (inside the training range)", fontsize=8.5, color=INK_2)
        ax.set_title(f"{'bus voltage' if level == 'bus' else 'branch loading'}: error rate of rows kept",
                     fontsize=9.5, color=INK, loc="left")
    axes[0].legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="center right", bbox_to_anchor=(1.0, 0.62))
    ax = axes[2]
    _style(ax)
    k = e3[e3.by == "k"].copy()
    k["kk"] = k.group.astype(int)
    for m, (lab, col, ls) in {"static_branch": ("fixed width per branch", SERIES[1], "-"),
                              "adaptive_isotonic": ("bands scaled by expected error", SERIES[0], "-"),
                              "static_global": ("one fixed width", INK_MUTED, (0, (3, 2)))}.items():
        c = k[k.method == m].sort_values("kk")
        ax.plot(c.kk, c.coverage, color=col, ls=ls, lw=1.8, marker=None if m == "static_global" else "o", ms=3.5,
                label=lab)
    ax.axhline(1 - alpha, color=INK_2, lw=0.9, ls=(0, (4, 3)))
    ax.set_xlabel("number of outages (bands set inside the training range)", fontsize=8.5, color=INK_2)
    ax.set_ylabel("hit rate of error bands for branch loading", fontsize=8.5, color=INK_2)
    ax.set_title(f"hit rate of {1 - alpha:.0%} error bands (target dashed)", fontsize=9.5, color=INK,
                 loc="left")
    ax.legend(frameon=False, fontsize=7.5, labelcolor=INK_2, loc="lower left")
    fig.tight_layout()
    return _save(fig, out_dir, grid, "error_rate_and_bands")


def a1_animation(grid: str = "texas2k", out_dir: str | Path | None = None) -> list[Path]:
    """Per-step mismatch animation for one N-1 and one high-k scenario (median final residual of its split)."""
    from .animate import residual_animation

    sb = io.read_table("static_bus", grid)
    xy = sb[["x", "y"]].to_numpy(np.float64)
    splits = ["test_1", "test_12"] if grid == "texas2k" else (["nk/k01", "nk/k10"] if grid == "case118" else ["test"])
    out = _out_dir(out_dir)
    paths = []
    for s in splits:
        sc = io.read_table("scenario", grid, s)
        pick = int(sc.iloc[int((sc.res_mean_final_mva - sc.res_mean_final_mva.median()).abs().argmin())].scenario)
        tr = io.read_traces(grid, s, np.array([pick]))
        bus = pd.read_parquet(io.results_dir(grid, s) / "bus.parquet", columns=["scenario", "bus", "struct_p", "struct_q"],
                              filters=[("scenario", "==", pick)]).sort_values("bus")
        k = int(sc.set_index("scenario").k.loc[pick])
        stem = f"mismatch_animation_{grid}_{_WORDS[k] if k < len(_WORDS) else k}_outage{'' if k == 1 else 's'}"
        paths += residual_animation(tr, pick, xy, str(out / stem),
                                    struct_p=bus.struct_p.to_numpy(), struct_q=bus.struct_q.to_numpy(),
                                    title=f"{_grid_name(grid)}, {k} outage{'s' if k > 1 else ''}, scenario {pick}")
    return paths


def f8_sensitivity(grid: str = "case118", out_dir: str | Path | None = None) -> list[Path]:
    """Gradient sensitivities: agreement of d loading / d Pd with finite-difference AC, GENCO automatic
    differentiation vs DC PTDF, per split: Spearman over buses and relative L2 error, one dot per (scenario, line)."""
    plt = mpl()
    p = io.results_dir(grid) / "e7_sensitivity.csv"
    if not p.exists():
        return []
    d = pd.read_csv(p)
    splits = list(dict.fromkeys(d.split))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.9))
    fig.patch.set_facecolor(SURFACE)
    rng = np.random.default_rng(0)
    for ax, (col, lab, log) in zip(axes, (("spearman", "Spearman vs finite-difference AC (over buses)", False),
                                          ("rel_l2", "relative L2 error vs finite-difference AC", True))):
        _style(ax)
        for si, sp in enumerate(splits):
            for off, src, colr, name in ((-0.17, "genco", SERIES[0], "GENCO autograd"),
                                         (0.17, "dc", SERIES[1], "DC PTDF")):
                v = d[(d.split == sp) & (d.source == src)][col].dropna().to_numpy()
                x = si + off + rng.uniform(-0.07, 0.07, v.size)
                ax.scatter(x, v, s=7, color=colr, alpha=0.45, linewidths=0, label=name if si == 0 else None)
                if v.size:
                    m = np.median(v)
                    ax.plot([si + off - 0.12, si + off + 0.12], [m, m], color=INK, lw=1.8, solid_capstyle="round")
        ax.set_xticks(range(len(splits)), [f"{int(s.split('k')[-1])} outage" + ("" if int(s.split('k')[-1]) == 1 else "s")
                                           for s in splits], fontsize=8.5)
        if log:
            ax.set_yscale("log")
        ax.set_ylabel(lab, fontsize=8.5, color=INK_2)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK_2, loc="lower left")
    axes[0].set_title(f"{_grid_name(grid)}: d loading / d Pd for the 10 most loaded lines per scenario", fontsize=9.5, color=INK,
                      loc="left")
    fig.text(0.01, -0.03, "One dot per (scenario, line); bars = medians. AC reference: numpy Newton-Raphson on datakit's "
             "network, central differences ±1 % Pd.", fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, grid, "sensitivity_against_newton_raphson")


def mismatch_against_last_step_change(grid: str = "texas2k", out_dir: str | Path | None = None,
                                      rows_inside: int = 300_000, rows_beyond: int = 40_000) -> list[Path]:
    """Headline: true |dVm| per PQ bus against the final mismatch and against the last-step voltage change.

    Rows = outage counts inside / beyond the training range; log-log hexbin (log counts) of a seeded row sample per
    split. Lines: error threshold tau (horizontal) and the fixed 90 %-recall threshold c of each check (vertical,
    from summary.last_step_check).
    """
    from ..config import grid_splits, k_fit, k_of_split

    plt = mpl()
    summ = io.read_summary()["last_step_check"][grid]["bus"]
    tau, k_max = summ["tau"], k_fit(grid)
    parts = []
    for i, s in enumerate(grid_splits(grid)):
        b = io.read_columns("bus", grid, s, ["k", "bus_type", "err_vm", "f_final", "nbmax_f_drift_vm"])
        b = b[(b.bus_type == "PQ") & b.err_vm.notna()]
        n = rows_inside if k_of_split(s) <= k_max else rows_beyond
        parts.append(b.sample(n=min(n, len(b)), random_state=i).drop(columns="bus_type"))
    d = pd.concat(parts, ignore_index=True)
    inside = d.k.to_numpy() <= k_max
    y = np.clip(d.err_vm.to_numpy(np.float64), 1e-7, None)
    cols = [("final_mismatch", "f_final", "final mismatch at the bus, |(dP, dQ)| [MVA]", 1e-4),
            ("last_step_vm_nbmax", "nbmax_f_drift_vm", "last-step voltage change, max over neighbouring buses [p.u.]",
             1e-7)]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 8.0), sharey=True)
    fig.patch.set_facecolor(SURFACE)
    cmap = mpl().cm.colors.LinearSegmentedColormap.from_list("hex", ("#e6eef9",) + SEQ_BLUE[3:])
    for r, (setname, m, rlab) in enumerate((("in_dist", inside, f"{k_max} or fewer outages (training range)"),
                                            ("shift_pooled", ~inside, f"more than {k_max} outages (beyond)"))):
        for c, (score, col, xlab, floor) in enumerate(cols):
            ax = axes[r, c]
            _style(ax)
            ax.grid(False)
            x = np.clip(d[col].to_numpy(np.float64)[m], floor, None)
            ax.hexbin(x, y[m], gridsize=60, xscale="log", yscale="log", bins="log", mincnt=1, cmap=cmap,
                      linewidths=0)
            v = summ[setname][score]
            ax.axhline(tau, color=INK, lw=1.0, ls=(0, (4, 2)))
            ax.axvline(v["threshold"], color=ACCENT, lw=1.6)
            ax.text(0.03, 0.97, f"AUROC {v['auroc']:.2f}\nflag right of the orange line:\n"
                    f"catches {100 * v['thr_recall']:.0f} % of wrong buses,\nflags {100 * v['thr_flag_rate']:.0f} % "
                    "of all buses", transform=ax.transAxes, va="top", fontsize=8, color=INK,
                    bbox={"boxstyle": "round,pad=0.35", "fc": SURFACE, "ec": GRID, "alpha": 0.9})
            if r == 1:
                ax.set_xlabel(xlab, fontsize=8.5, color=INK_2)
            if c == 0:
                ax.set_ylabel(f"true voltage error |V - V_AC| [p.u.]\n{rlab}", fontsize=8.5, color=INK_2)
        axes[r, 0].text(1.0, tau * 1.25, f"error threshold {tau} p.u.", transform=axes[r, 0].get_yaxis_transform(),
                        ha="right", fontsize=7.5, color=INK)
    axes[0, 0].set_title("final mismatch: wrong and correct buses overlap", fontsize=9.5, color=INK, loc="left")
    axes[0, 1].set_title("last-step change: wrong buses sit to the right", fontsize=9.5, color=INK, loc="left")
    fig.text(0.01, -0.02, f"{grid}: one cell per (bus, scenario), PQ buses, colour = log count of a seeded sample "
             f"({rows_inside:,} rows per split inside, {rows_beyond:,} beyond). Orange line: threshold chosen once, "
             f"inside the training range, to catch 90 % of wrong buses; kept fixed beyond. Values below 1e-7 "
             "(float32 resolution) are drawn at 1e-7.", fontsize=7,
             color=INK_MUTED, wrap=True)
    fig.tight_layout()
    return _save(fig, out_dir, grid, "mismatch_against_last_step_change")


def newton_step_and_conditioning(grid: str = "texas2k", out_dir: str | Path | None = None,
                                 max_points: int = 60_000) -> list[Path]:
    """True |dVm| per PQ bus against (a) the one-Newton-step estimate and (b) the final mismatch, coloured by the
    scenario's sigma_min(J) quintile; (c) median (max true |dVm| / max |g|) per sigma_min quintile and per outage
    count. From results/<grid>/newton_{bus.parquet,scenario.csv,conditioning.csv}."""
    plt = mpl()
    d = io.results_dir(grid)
    B = pd.read_parquet(d / "newton_bus.parquet")
    S = pd.read_csv(d / "newton_scenario.csv")
    C = pd.read_csv(d / "newton_conditioning.csv")
    S["quint"] = pd.qcut(S.sigma_min, 5, labels=False) + 1
    B = B.merge(S[["split", "scenario", "quint"]], on=["split", "scenario"], how="left")
    if len(B) > max_points:
        B = B.sample(n=max_points, random_state=0)
    tau = io.read_summary()["e2"][grid]["bus"]["headline_tau"]
    y = np.clip(B.err_vm.to_numpy(np.float64), 1e-7, None)
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.3), gridspec_kw={"width_ratios": [1, 1, 1.05]})
    fig.patch.set_facecolor(SURFACE)
    for ax in axes:
        _style(ax)
    cmap = mpl().cm.colors.LinearSegmentedColormap.from_list("hex", ("#e6eef9",) + SEQ_BLUE[3:])
    ax = axes[0]
    ax.grid(False)
    x = np.clip(B.newton_dvm.to_numpy(np.float64), 1e-7, None)
    ax.hexbin(x, y, gridsize=55, xscale="log", yscale="log", bins="log", mincnt=1, cmap=cmap, linewidths=0)
    ax.plot([1e-7, 1], [1e-7, 1], color=INK, lw=0.9, ls=(0, (4, 2)))
    ax.axhline(tau, color=ACCENT, lw=1.2)
    ax.set_xlim(1e-7, 0.3)
    ax.set_ylim(1e-7, 0.3)
    ax.set_xlabel("one Newton step from GENCO's solution, |dV| [p.u.]", fontsize=8.5, color=INK_2)
    ax.set_ylabel("true voltage error |V - V_AC| [p.u.]", fontsize=8.5, color=INK_2)
    ax.set_title("(a) one Newton step recovers the error", fontsize=9.5, color=INK, loc="left")
    ax = axes[1]
    ax.grid(False)
    ax.hexbin(np.clip(B.f_final.to_numpy(np.float64), 1e-4, None), y, gridsize=55, xscale="log", yscale="log",
              bins="log", mincnt=1, cmap=cmap, linewidths=0)
    ax.axhline(tau, color=ACCENT, lw=1.2)
    ax.set_ylim(1e-7, 0.3)
    ax.set_xlabel("final mismatch at the bus [MVA]", fontsize=8.5, color=INK_2)
    ax.set_title("(b) the same error against the final mismatch", fontsize=9.5, color=INK, loc="left")
    ax = axes[2]
    k = C[C.by == "k"].sort_values("q")
    q = C[C.by == "sigma_min_quintile"].sort_values("q")
    ax.plot(k.q, k.ratio_median, marker="o", color=SERIES[0], lw=2)
    ax.set_xlabel("number of outages", fontsize=8.5, color=INK_2)
    ax.set_ylabel("median of max |V error| / max |mismatch| per scenario", fontsize=8.5, color=INK_2)
    ax.set_title("(c) voltage error per unit of mismatch", fontsize=9.5, color=INK, loc="left")
    ax.set_ylim(0, None)
    fig.text(0.01, -0.03, f"{grid}: {S.scenario.size:,} scenarios ({S.groupby('k').size().iloc[0]} per outage count), "
             f"PQ buses ({len(B):,} shown). Newton step: -J(x)^-1 g(x) at GENCO's solution with PV/REF setpoints, on each "
             f"scenario's own network; orange line: error threshold. (c) mismatch in p.u. on 100 MVA; sigma_min(J) varies "
             f"only from {q.sigma_min_median.min():.3f} to {q.sigma_min_median.max():.3f} (quintile medians).",
             fontsize=7, color=INK_MUTED)
    fig.tight_layout()
    return _save(fig, out_dir, grid, "newton_step_and_conditioning")


FIGURES = {"mismatch_against_last_step_change": mismatch_against_last_step_change,
           "mismatch_by_step": f2_anatomy, "error_check_comparison": f3_auroc,
           "error_rate_and_bands": f4_risk_coverage, "ac_resolve_selection": f5_hybrid,
           "stress_agreement": f6_stress_agreement, "stress_and_accuracy_map": f1_joint_map,
           "stress_disagreement_by_group": f7_disagreement, "newton_step_and_conditioning": newton_step_and_conditioning,
           "sensitivity_against_newton_raphson": f8_sensitivity, "mismatch_animation": a1_animation}
# old figure ids, kept until the release so that existing commands keep working
ALIASES = {"F1": "stress_and_accuracy_map", "F2": "mismatch_by_step", "F3": "error_check_comparison",
           "F4": "error_rate_and_bands", "F5": "ac_resolve_selection", "F6": "stress_agreement",
           "F7": "stress_disagreement_by_group", "F8": "sensitivity_against_newton_raphson",
           "A1": "mismatch_animation", "headline": "mismatch_against_last_step_change",
           "newton": "newton_step_and_conditioning"}
