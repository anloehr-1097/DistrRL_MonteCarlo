
import numpy as np

def assert_probs_distr(probs: np.ndarray) -> None:
    assert np.isclose(np.sum(probs), 1), "Probs do not sum to 1."

