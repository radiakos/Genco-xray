# One target per stage. GRID=texas2k|case118|case118_small|case118_seed1|case118_seed42|synthetic
GRID ?= case118
.PHONY: test download sanity-checks capture-intermediate-solutions build-features error-check ac-resolve-selection \
        stress-maps stress-and-accuracy-map gradient-sensitivities last-step-check large-loading-errors newton-check \
        scenario-check-scores uncertainty-baselines figures synthetic all

test: ; pytest -q
download: ; python scripts/download_checkpoints_and_data.py --grid $(GRID)
sanity-checks: ; python scripts/capture_intermediate_solutions.py --grid $(GRID) --e0-only
capture-intermediate-solutions: ; python scripts/capture_intermediate_solutions.py --grid $(GRID)
build-features: ; python scripts/build_features.py --grid $(GRID)
error-check: ; python scripts/error_check.py --grid $(GRID)
ac-resolve-selection: ; python scripts/ac_resolve_selection.py --grid $(GRID)
stress-maps: ; python scripts/stress_maps.py --grid $(GRID)
stress-and-accuracy-map: ; python scripts/stress_and_accuracy_map.py --grid $(GRID) && python scripts/make_figures.py --grid $(GRID)
gradient-sensitivities: ; python scripts/gradient_sensitivities.py --grid case118
last-step-check: ; python scripts/last_step_check.py --grid $(GRID) && python scripts/make_figures.py --grid $(GRID) --only mismatch_against_last_step_change
large-loading-errors: ; python scripts/large_loading_errors.py --grid $(GRID)
newton-check: ; python scripts/newton_check.py --grid $(GRID) && python scripts/make_figures.py --grid $(GRID) --only newton_step_and_conditioning
scenario-check-scores: ; python scripts/scenario_check_scores.py --grid $(GRID)
uncertainty-baselines: ; python scripts/uncertainty_baselines.py --grid $(GRID)
figures: ; python scripts/make_figures.py --grid $(GRID)
synthetic: ; python scripts/build_features.py --grid synthetic --synthetic && \
             for s in error_check ac_resolve_selection stress_maps stress_and_accuracy_map; do python scripts/$$s.py --grid synthetic; done
all: capture-intermediate-solutions build-features error-check ac-resolve-selection stress-maps stress-and-accuracy-map
