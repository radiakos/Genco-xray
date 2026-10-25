"""M0: restricting the checkpoint's train/val/test split to a downloaded prefix of the dataset."""

import json

import pandas as pd
import pytest
import torch

from gencoxray.checkpoints import _partition_patterns, write_split_files


def _hive_bus_table(raw, scenarios, per_partition=3):
    for p in range(0, len(scenarios), per_partition):
        d = raw / "bus_data.parquet" / f"scenario_partition={p // per_partition}"
        d.mkdir(parents=True)
        pd.DataFrame({"scenario": scenarios[p:p + per_partition], "bus": 0}).to_parquet(d / "x.parquet")


def test_split_restricted_to_present_scenarios(tmp_path):
    raw = tmp_path / "raw"
    _hive_bus_table(raw, list(range(6)))
    splits = {"train": [2, 3, 7, 9], "val": [4, 8], "test": [5, 1, 0, 6]}
    (tmp_path / "s.json").write_text(json.dumps(splits))
    counts = write_split_files(tmp_path / "s.json", raw, tmp_path / "splits")
    assert counts == {"train": 2, "val": 1, "test": 3}
    assert torch.load(tmp_path / "splits" / "test.pt").tolist() == [0, 1, 5]
    assert torch.load(tmp_path / "splits" / "train.pt").tolist() == [2, 3]


def test_split_must_cover_every_present_scenario(tmp_path):
    raw = tmp_path / "raw"
    _hive_bus_table(raw, list(range(4)))
    (tmp_path / "s.json").write_text(json.dumps({"train": [0, 1], "val": [], "test": [2]}))
    with pytest.raises(RuntimeError, match="cover 3 of 4"):
        write_split_files(tmp_path / "s.json", raw, tmp_path / "splits")


def test_partition_patterns():
    pats = _partition_patterns("data/test_1/X/raw/", 2)
    assert "data/test_1/X/raw/bus_data.parquet/scenario_partition=1/*" in pats
    assert not any("scenario_partition=2" in p for p in pats)
    assert pats[-1] == "data/test_1/X/raw/n_scenarios.txt"
    assert "branch_data.parquet/scenario_partition=*/*" in _partition_patterns("", None)
