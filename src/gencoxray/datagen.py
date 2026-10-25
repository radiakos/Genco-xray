"""Generate IEEE 118-bus contingency test sets (exactly k outages) with gridfm-datakit.

Template: configs/datagen/case118_nk.yaml with placeholders __K__, __KK__ (zero-padded), __ONEHOT__, __SEED__.
Output must end up as ``data/case118/nk/kXX/case118_ieee/raw/*.parquet`` so graphkit can read each k as a split
(link the datakit output directory there if it writes elsewhere).
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import yaml

from .config import REPO_ROOT, resolve  # noqa: F401 — REPO_ROOT is re-exported (tests)

log = logging.getLogger(__name__)


def fill_template(text: str, k: int, seed: int) -> dict:
    """Return the datakit config dict for exactly-k outages (outage_count_probabilities one-hot at index k)."""
    onehot = [0.0] * k + [1.0]
    text = text.replace("__ONEHOT__", "[" + ", ".join(str(v) for v in onehot) + "]")
    text = text.replace("__KK__", f"{k:02d}").replace("__K__", str(k))
    text = text.replace("__SEED__", str(seed + k))
    cfg = yaml.safe_load(text)
    cfg["topology_perturbation"]["k"] = int(k)
    return cfg


def write_nk_configs(template: str | Path, ks: list[int], out_dir: str | Path, seed: int = 0) -> list[Path]:
    """Write one datakit YAML per k (``case118_nk_k01.yaml`` …) and return their paths."""
    tpl, out = resolve(template), resolve(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = tpl.read_text()
    paths = []
    for k in ks:
        p = out / f"case118_nk_k{k:02d}.yaml"
        p.write_text(yaml.safe_dump(fill_template(text, k, seed), sort_keys=False))
        paths.append(p)
    return paths


def run_datakit(config_path: str | Path) -> Path:
    """Run ``gridfm_datakit generate <config>`` and return its ``settings.data_dir``."""
    cfg = yaml.safe_load(Path(config_path).read_text())
    log.info("datakit generate %s", config_path)
    subprocess.run(["gridfm_datakit", "generate", str(config_path)], check=True)
    return Path(cfg["settings"]["data_dir"])
