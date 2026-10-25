"""Already passing: the synthetic generator emits the real schemas and the planted structure."""

import numpy as np

from gencoxray import schemas


def test_tables_validate(synth):
    schemas.validate(synth.bus, schemas.BUS)
    schemas.validate(synth.branch, schemas.BRANCH)
    schemas.validate(synth.scenario, schemas.SCENARIO)
    schemas.validate(synth.static_bus, schemas.STATIC_BUS)
    schemas.validate(synth.static_branch, schemas.STATIC_BRANCH)
    schemas.validate_traces(synth.traces)


def test_shapes_consistent(synth):
    S, T, B = synth.traces["res_p"].shape
    E2 = synth.traces["loading"].shape[2]
    assert T == 12 and len(synth.bus) == S * B and len(synth.static_bus) == B
    assert E2 == 2 * len(synth.static_branch) and len(synth.branch) == S * len(synth.static_branch)


def test_structural_zeros_respected(synth):
    bt = synth.static_bus.bus_type.to_numpy().astype(str)
    sp, sq = schemas.structural_zero_mask(bt)
    assert np.all(synth.traces["res_q"][:, :, sq] == 0)
    assert np.all(synth.traces["res_p"][:, :, sp] == 0)


def test_planted_region_is_harder(synth):
    hard = synth.truth["hard"]
    assert hard[:, synth.truth["region_buses"]].mean() > 2 * hard.mean() * 0.9


def test_load_scenario_groups(synth):
    assert (synth.scenario.load_scenario == synth.scenario.scenario // 4).all()
    assert synth.bus.groupby("scenario").load_scenario.nunique().max() == 1


def test_final_step_matches_tables(synth):
    S, T, B = synth.traces["res_p"].shape
    np.testing.assert_allclose(synth.bus.res_p_final_mw.to_numpy().reshape(S, B), synth.traces["res_p"][:, -1])
    np.testing.assert_allclose(synth.bus.vm_pred.to_numpy().reshape(S, B), synth.traces["vm"][:, -1], atol=1e-6)
