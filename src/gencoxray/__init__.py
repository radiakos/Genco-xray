"""GENCO X-Ray: correction-trajectory trust signals, grid stress maps and their joint Stress × Trust map.

Torch-dependent modules (checkpoints, trace, runner, sensitivity) are imported lazily, so the
analysis stack (features, trust, conformal, hybrid, stress, ranking, joint, viz) runs without torch.
"""

__version__ = "0.1.0"
