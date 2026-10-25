"""Acceptance tests for the scenario-level scores of scripts/scenario_check_scores.py (no data needed)."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_spec = importlib.util.spec_from_file_location(
    "scenario_check_scores", Path(__file__).parents[1] / "scripts" / "scenario_check_scores.py")
scs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(scs)

THR = {"vm": 0.01, "mis_bus": 1.0, "ld": 0.02, "mis_br": 2.0}
TAU = {"bus": 0.005, "branch": 0.02}


@pytest.fixture
def tables():
    bus = pd.DataFrame({
        "scenario": [0, 0, 0, 1, 1, 1], "bus_type": ["PQ", "PQ", "PV", "PQ", "PQ", "PQ"],
        "nbmax_f_drift_vm": [0.02, 0.001, 0.5, 0.001, 0.002, 0.003], "f_final": [0.1, 0.2, 9.0, 0.1, 3.0, 0.2],
        "err_vm": [0.0, 0.01, 0.0, 0.0, 0.0, 0.0]})
    br = pd.DataFrame({
        "scenario": [0, 0, 1, 1], "in_service": [True, True, True, False],
        "loading_pred": [1.2, 0.5, 0.9, np.nan], "g_drift_tail_std": [0.001, 0.001, 0.001, np.nan],
        "g_endpoint_max_f_final": [0.5, 0.5, 0.5, np.nan], "err_loading": [0.0, 0.0, 0.0, np.nan]})
    return bus, br


def test_counts_and_flags(tables):
    out = scs.table_scores(*tables, THR, TAU, 1.0)
    assert out.loc[0, "n_vm_flag"] == 1 and out.loc[1, "n_vm_flag"] == 0  # PV bus 0.5 is excluded
    assert out.loc[0, "max_mis_bus"] == pytest.approx(0.2)  # PV mismatch 9.0 excluded
    assert bool(out.loc[0, "any_wrong_bus"]) and not bool(out.loc[1, "any_wrong_bus"])
    assert bool(out.loc[0, "flag_last_only"]) and not bool(out.loc[1, "flag_last_only"])
    # scenario 1 passes the last-step rule but its final mismatch 3.0 >= 1.0 trips the recommendation
    assert bool(out.loc[1, "flag_recommendation"])


def test_recommendation_score_is_scale_free(tables):
    out = scs.table_scores(*tables, THR, TAU, 1.0)
    assert out.loc[0, "rec_score"] == pytest.approx(max(0.02 / 0.01, 0.2 / 1.0, 0.001 / 0.02, 0.5 / 2.0))
    assert out.loc[1, "rec_score"] == pytest.approx(3.0 / 1.0)


def test_last_step_margin_flip(tables):
    out = scs.table_scores(*tables, THR, TAU, 1.0)
    # scenario 0 is predicted overloaded through branch 0 (L = 1.2): z = 0.2 / 0.001 = 200; the out-of-service branch
    # of scenario 1 is ignored, its only branch at 0.9 gives -z = 0.1 / 0.001 = 100
    assert out.loc[0, "z_last_flip"] == pytest.approx(200.0)
    assert out.loc[1, "z_last_flip"] == pytest.approx(100.0)
