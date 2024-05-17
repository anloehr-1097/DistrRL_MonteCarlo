"""Simple testing."""

import numpy as np
from typing import Tuple
from ...preliminary_tests import categorical_projection


def test_cat_proj():
    """Test categorical projection."""
    particles: np.ndarray = np.linspace(0, 10, 11)
    values: np.ndarray = np.linspace(-0.5, 10.5, 12)
    print(particles)
    print(values)
    probs: np.ndarray = np.ones(np.size(values))
    probs = probs / np.sum(probs)
    # probs: np.ndarray = np.random.random(20)
    # probs = probs / np.sum(probs)
    print(probs)
    # assert np.isclose(np.sum(probs), 1), "Probs do not sum to 1." 
    new_distr: Tuple[np.ndarray, np.ndarray] = categorical_projection(
        (values, probs), particles)

    assert np.isclose(np.sum(new_distr[1]), 1), "not a prob distribution after projection."
    return None


if __name__ == "__main__":
    test_cat_proj()
    print("Everything passed")
