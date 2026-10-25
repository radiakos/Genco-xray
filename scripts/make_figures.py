"""Produce the submission figures from results/ only (gencoxray.viz.figures).

    python scripts/make_figures.py --grid texas2k
    python scripts/make_figures.py --grid texas2k --only ac_resolve_selection stress_agreement

Figure names: mismatch_against_last_step_change, mismatch_by_step, error_check_comparison, error_rate_and_bands,
ac_resolve_selection, stress_agreement, stress_and_accuracy_map, stress_disagreement_by_group,
newton_step_and_conditioning, sensitivity_against_newton_raphson, mismatch_animation. Earlier figure ids are accepted
as aliases. Figures whose inputs the grid lacks (for example the Newton figure on a grid without
newton_check output) are skipped with a message.
"""

import logging

from gencoxray import cli
from gencoxray.viz.figures import ALIASES, FIGURES

log = logging.getLogger("figures")

if __name__ == "__main__":
    ap = cli.parser(__doc__)
    ap.add_argument("--grid", default="texas2k")
    ap.add_argument("--only", nargs="*", default=None, choices=list(FIGURES) + list(ALIASES),
                    help="subset of figures, by name or old id")
    a = ap.parse_args()
    cli.setup_logging()
    for name in [ALIASES.get(n, n) for n in a.only] if a.only else FIGURES:
        try:
            paths = FIGURES[name](a.grid)
        except (FileNotFoundError, KeyError) as e:
            log.warning("%s skipped for %s: missing input %s", name, a.grid, e)
            continue
        for p in paths:
            log.info("%s -> %s", name, p)
