"""Already passing: schema machinery (implemented)."""

import numpy as np
import pandas as pd
import pytest

from gencoxray import schemas


def test_structural_zero_mask_pf():
    sp, sq = schemas.structural_zero_mask(np.array(["PQ", "PV", "REF"]))
    assert sp.tolist() == [False, False, True]
    assert sq.tolist() == [False, True, True]


def test_structural_zero_mask_opf():
    sp, sq = schemas.structural_zero_mask(np.array(["PQ", "PV", "REF"]), task="OptimalPowerFlow")
    assert sp.tolist() == [False, False, False]
    assert sq.tolist() == [False, True, True]


def test_validate_catches_missing_and_dtype():
    df = pd.DataFrame({"branch": np.arange(3, dtype=np.int32), "from_bus": [0, 1, 2], "to_bus": [1.0, 2.0, 0.0]})
    with pytest.raises(schemas.SchemaError) as e:
        schemas.validate(df, schemas.STATIC_BRANCH)
    msg = str(e.value)
    assert "rate_a_mva" in msg and "to_bus" in msg


def test_feature_name_sets_consistent():
    assert len(schemas.BUS_NB_FEATURES) == len(schemas.BUS_FEATURES)
    assert all(f"g_endpoint_max_{f}" in schemas.BRANCH_FEATURES for f in schemas.BUS_FEATURES)
    bus_cols = {c.name for c in schemas.BUS}
    assert set(schemas.BUS_FEATURES) <= bus_cols and set(schemas.BUS_NB_FEATURES) <= bus_cols
