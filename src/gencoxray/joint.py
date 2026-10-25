"""Stress and accuracy map, and the test that stressed error-prone branches disagree more with AC."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import ranking, stress

QUADRANTS = ("stressed_untrusted", "stressed_trusted", "calm_untrusted", "calm_trusted")
STRESS_SOURCES = {"genco": "loading_pred", "ac": "loading_true", "dc": "loading_dc"}


def element_trust_parts(branch_df: pd.DataFrame, stressed_loading: float = 0.9) -> pd.DataFrame:
    """Per branch sums for :func:`element_trust`, so large tables can be aggregated split by split:
    risk_stressed, n_stressed (in service and loading_pred > stressed_loading), risk_all, n_all (in service)."""
    on = branch_df.in_service.to_numpy() & np.isfinite(branch_df.branch_risk.to_numpy())
    d = branch_df.loc[on, ["branch", "loading_pred", "branch_risk"]]
    st = d.loading_pred.to_numpy() > stressed_loading
    g = d.groupby("branch")
    out = pd.DataFrame({"risk_all": g.branch_risk.sum().astype(np.float64), "n_all": g.size()})
    gs = d[st].groupby("branch")
    out["risk_stressed"] = gs.branch_risk.sum().astype(np.float64).reindex(out.index).fillna(0.0)
    out["n_stressed"] = gs.size().reindex(out.index).fillna(0).astype(np.int64)
    return out


def trust_from_parts(parts: pd.DataFrame) -> pd.DataFrame:
    """trust = -mean risk over stressed scenarios, falling back to all in-service scenarios; plus the counts."""
    p = parts.groupby(level=0).sum()
    stressed = p.n_stressed > 0
    mean = np.where(stressed, p.risk_stressed / p.n_stressed.clip(lower=1), p.risk_all / p.n_all.clip(lower=1))
    return pd.DataFrame({"trust": -mean, "n_stressed": p.n_stressed, "n_all": p.n_all,
                         "trust_from_stressed": stressed}, index=p.index)


def element_trust(branch_df: pd.DataFrame, stressed_loading: float = 0.9) -> pd.Series:
    """Per branch: trust = -mean(``branch_risk``) over in-service scenarios where loading_pred > stressed_loading
    (fallback: all in-service scenarios if none qualify). Higher = more trustworthy. Index = branch.

    Element-level risk (not scenario trust) is required. Scenario trust is too coarse, because stressed elements
    occur mostly in high-k scenarios, which makes every stressed element look 'untrusted'."""
    return trust_from_parts(element_trust_parts(branch_df, stressed_loading)).trust.rename("trust")


def quadrants(stress: pd.Series, trust: pd.Series, stress_q: float = 0.9, trust_q: float = 1 / 3) -> pd.Series:
    """Categorical per element: stressed = stress >= quantile(stress_q); untrusted = trust <= quantile(trust_q).
    Categories exactly ``QUADRANTS``. Elements with NaN stress or trust -> NaN."""
    s, t = stress.align(trust, join="outer")
    ok = s.notna() & t.notna()
    stressed = s >= s[ok].quantile(stress_q)
    untrusted = t <= t[ok].quantile(trust_q)
    lab = np.where(stressed, np.where(untrusted, "stressed_untrusted", "stressed_trusted"),
                   np.where(untrusted, "calm_untrusted", "calm_trusted")).astype(object)
    lab[~ok.to_numpy()] = None
    return pd.Series(pd.Categorical(lab, categories=QUADRANTS), index=s.index, name="quadrant")


def disagreement(stress_genco: pd.Series, stress_ac: pd.Series, kind: str = "abs") -> pd.Series:
    """Per element: 'abs' = |stress_genco - stress_ac| (primary, in metric units);
    'rank' = ranking.rank_disagreement (secondary; insensitive within the top set, so reported only)."""
    if kind == "abs":
        g, a = stress_genco.align(stress_ac, join="inner")
        return (g - a).abs().rename("disagreement")
    if kind == "rank":
        return ranking.rank_disagreement(stress_genco, stress_ac).rename("disagreement_rank")
    raise ValueError(kind)


def h4_test(disagreement: pd.Series, quadrant: pd.Series) -> dict:
    """Mann-Whitney U (one-sided: stressed_untrusted > stressed_trusted) on disagreement.
    Returns dict(u, p_value, rank_biserial, n_untrusted, n_trusted, median_untrusted, median_trusted).
    rank_biserial = 2U / (n_u n_t) - 1 (> 0: untrusted elements disagree more)."""
    from scipy.stats import mannwhitneyu

    d, q = disagreement.align(quadrant, join="inner")
    a = d[(q == "stressed_untrusted").to_numpy()].dropna().to_numpy()
    b = d[(q == "stressed_trusted").to_numpy()].dropna().to_numpy()
    out = {"n_untrusted": int(a.size), "n_trusted": int(b.size),
           "median_untrusted": float(np.median(a)) if a.size else float("nan"),
           "median_trusted": float(np.median(b)) if b.size else float("nan")}
    if a.size == 0 or b.size == 0:
        return out | {"u": float("nan"), "p_value": float("nan"), "rank_biserial": float("nan")}
    res = mannwhitneyu(a, b, alternative="greater")
    return out | {"u": float(res.statistic), "p_value": float(res.pvalue),
                  "rank_biserial": float(2 * res.statistic / (a.size * b.size) - 1)}


def assemble(stress_maps: dict[str, pd.Series], trust: pd.Series, static_branch: pd.DataFrame, *,
             stress_q: float = 0.9, trust_q: float = 1 / 3) -> pd.DataFrame:
    """Joint table from per-branch stress maps {'genco', 'ac', ['dc']} and element trust (index = branch)."""
    sb = static_branch.set_index("branch").sort_index()
    jt = pd.DataFrame(index=sb.index)
    for src, s in stress_maps.items():
        jt[f"stress_{src}"] = s.reindex(jt.index)
    if "stress_dc" not in jt:
        jt["stress_dc"] = np.nan
    jt["trust"] = trust.reindex(jt.index)
    jt["quadrant"] = quadrants(jt.stress_genco, jt.trust, stress_q, trust_q)
    jt["disagreement"] = disagreement(jt.stress_genco, jt.stress_ac, "abs")
    jt["disagreement_rank"] = disagreement(jt.stress_genco, jt.stress_ac, "rank").reindex(jt.index)
    jt["from_bus"], jt["to_bus"] = sb.from_bus, sb.to_bus
    jt.index.name = "branch"
    return jt


def joint_table(branch_df: pd.DataFrame, scenario_df: pd.DataFrame, static_branch: pd.DataFrame,
                metric: str = "p_overload", *, k_range: tuple[int, int] | None = None, stressed_loading: float = 0.9,
                stress_q: float = 0.9, trust_q: float = 1 / 3, cvar_q: float = 0.95) -> pd.DataFrame:
    """One row per branch: stress_genco, stress_ac, stress_dc, trust (from :func:`element_trust`), quadrant,
    disagreement (kind='abs'), disagreement_rank, from_bus, to_bus. ``scenario_df`` filters by contingency order
    (``k_range=(lo, hi)``). This is the table behind the stress and accuracy map figure."""
    br = branch_df
    if k_range is not None:
        lo, hi = k_range
        keep = scenario_df.scenario[(scenario_df.k >= lo) & (scenario_df.k <= hi)]
        br = br[br.scenario.isin(keep)]
    maps = {src: stress.branch_stress(br, col, cvar_q=cvar_q)[metric] for src, col in STRESS_SOURCES.items()
            if col in br and br[col].notna().any()}
    trust = element_trust(br, stressed_loading)
    return assemble(maps, trust, static_branch, stress_q=stress_q, trust_q=trust_q)
