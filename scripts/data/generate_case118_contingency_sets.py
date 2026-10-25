"""Generate IEEE 118-bus contingency test sets with exactly k = 1..10 outages with gridfm-datakit (needs Julia:
`gridfm_datakit setup_pm`).

Each set lands in data/case118/data/nk/kXX/case118_ieee/raw/ (= graphkit split "nk/kXX") with a splits/ folder that
puts every scenario in the test split (the checkpoint never saw these topologies). On Windows run with
JULIA_PKG_SERVER="" and see CLAUDE.md ("datakit on Windows") for the one-time PGLib download / corrected-case
workaround.

    python scripts/data/generate_case118_contingency_sets.py                 # generate k = 1..10, then write split files
    python scripts/data/generate_case118_contingency_sets.py --splits-only   # (re)write split files for existing sets
"""

import argparse
from pathlib import Path

from gencoxray.checkpoints import write_all_test_split
from gencoxray.config import REPO_ROOT, load_config
from gencoxray.datagen import run_datakit, write_nk_configs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ks", type=int, nargs="+", default=list(range(1, 11)))
    ap.add_argument("--splits-only", action="store_true")
    a = ap.parse_args()
    exp = load_config()
    cfgs = write_nk_configs("configs/datagen/case118_nk.yaml", a.ks, "configs/datagen/generated", seed=exp.seed)
    for k, cfg in zip(a.ks, cfgs):
        root = REPO_ROOT / f"data/case118/data/nk/k{k:02d}"
        if not a.splits_only:
            root = Path(run_datakit(cfg))
            root = root if root.is_absolute() else REPO_ROOT / root
        n = write_all_test_split(root / "case118_ieee" / "raw", root / "splits")
        print(f"k={k:02d}: {n} scenarios -> {root}")
