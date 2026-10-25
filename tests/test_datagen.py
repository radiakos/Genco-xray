"""Tests for datagen.py (implemented; no datakit needed)."""

import yaml

from gencoxray import datagen


def test_fill_template_onehot_and_paths():
    text = open(datagen.REPO_ROOT / "configs/datagen/case118_nk.yaml").read()
    cfg = datagen.fill_template(text, 3, seed=10)
    tp = cfg["topology_perturbation"]
    assert tp["k"] == 3 and tp["outage_count_probabilities"] == [0.0, 0.0, 0.0, 1.0]
    assert "k03" in cfg["settings"]["data_dir"] and cfg["settings"]["seed"] == 13
    assert cfg["settings"]["mode"] == "pf" and cfg["settings"]["include_dc_res"] is True


def test_write_nk_configs(tmp_path):
    paths = datagen.write_nk_configs("configs/datagen/case118_nk.yaml", [1, 10], tmp_path)
    assert [p.name for p in paths] == ["case118_nk_k01.yaml", "case118_nk_k10.yaml"]
    assert yaml.safe_load(paths[1].read_text())["topology_perturbation"]["k"] == 10
