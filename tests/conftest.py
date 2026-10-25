import numpy as np
import pytest

from gencoxray.synthetic import make_dataset


@pytest.fixture(scope="session")
def synth():
    """Synthetic dataset (features NaN, trust NaN)."""
    return make_dataset(seed=0)


@pytest.fixture(scope="session")
def synth_oracle():
    """Synthetic dataset with noisy oracle branch_risk / trust_score (lets WS-C and hybrid tests run without WS-B)."""
    return make_dataset(seed=1, n_scenarios=800, fill_oracle_trust=True)


@pytest.fixture
def rng():
    return np.random.default_rng(0)
