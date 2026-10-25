"""End-to-end contract on a real checkpoint (WS-A, M1/M2). Opt-in on the GPU box after M0:

    GENCOXRAY_GPU_TESTS=1 pytest -q tests/test_runner_contract.py
"""

import os

import numpy as np
import pytest

pytestmark = [pytest.mark.gpu, pytest.mark.skipif(os.environ.get("GENCOXRAY_GPU_TESTS") != "1",
                                                  reason="set GENCOXRAY_GPU_TESTS=1 to run")]


def test_e0_and_tables_case118():
    from gencoxray import schemas
    from gencoxray.checkpoints import download, load_genco
    from gencoxray.config import load_config, load_grid
    from gencoxray.runner import e0_sanity, run_split

    exp, grid = load_config(), load_grid("case118")
    loaded = load_genco(grid, download(grid, exp.data_dir), split="test", device=exp.device)
    assert loaded.n_steps == exp.trace.n_steps_expected
    assert loaded.base_mva > 0  # fitted (95th pct of powers), generally != 100
    e0 = e0_sanity(loaded, n_scenarios=64)
    assert e0["passed"], e0
    res = run_split(loaded, grid, exp, "test", max_scenarios=64, write=False)
    schemas.validate(res.bus, schemas.BUS)
    schemas.validate(res.branch, schemas.BRANCH)
    schemas.validate(res.scenario, schemas.SCENARIO)
    assert res.bus.vm_pred.between(0.5, 1.5).all()
    assert (res.bus.res_final_mva >= 0).all()
    assert 0 < np.nanmedian(res.branch.loading_true) < 2
    assert (res.scenario.load_scenario >= 0).all()
