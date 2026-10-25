"""Acceptance tests for joint.py (WS-C). Uses oracle trust, so it is independent of WS-B."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import joint


def test_quadrants_counts():
    stress = pd.Series(np.arange(100, dtype=float))
    trust = pd.Series(np.tile([0.0, 1.0], 50))
    q = joint.quadrants(stress, trust, stress_q=0.9, trust_q=0.5)
    assert set(q.cat.categories) == set(joint.QUADRANTS)
    assert (q == "stressed_untrusted").sum() == 5 and (q == "stressed_trusted").sum() == 5


def test_disagreement_abs():
    a, b = pd.Series([0.1, 0.5]), pd.Series([0.2, 0.5])
    np.testing.assert_allclose(joint.disagreement(a, b, kind="abs"), [0.1, 0.0])


def test_element_trust_uses_branch_risk():
    br = pd.DataFrame({"scenario": np.array([0, 1, 0, 1], dtype=np.int64), "branch": np.array([0, 0, 1, 1], dtype=np.int32),
                       "in_service": [True] * 4, "loading_pred": np.array([0.95, 0.5, 0.5, 0.5], dtype=np.float32),
                       "branch_risk": np.array([2.0, 0.0, 1.0, 3.0], dtype=np.float32)})
    t = joint.element_trust(br, stressed_loading=0.9)
    assert t.loc[0] == pytest.approx(-2.0)  # only the stressed scenario counts
    assert t.loc[1] == pytest.approx(-2.0)  # fallback: mean over all scenarios


def test_h4_detects_planted_region(synth_oracle):
    jt = joint.joint_table(synth_oracle.branch, synth_oracle.scenario, synth_oracle.static_branch)
    assert {"stress_genco", "stress_ac", "trust", "quadrant", "disagreement", "disagreement_rank"} <= set(jt.columns)
    res = joint.h4_test(jt.disagreement, jt.quadrant)
    assert res["median_untrusted"] > res["median_trusted"]
    assert res["p_value"] < 0.05
