"""Simple testing."""

import numpy as np
from typing import Tuple
import unittest
from src.preliminary_tests import categorical_projection

DEBUG: bool = False


class TestCategoricalProjection(unittest.TestCase):
    """Test code for categorical projection."""

    def test_cat_proj(self):
        """Test categorical projection."""
        print("Testing categorical projection...")

        # particles: np.ndarray = np.linspace(0, 3, 4)
        # values: np.ndarray = np.ones(shape=(4, )) / particles.size
        values: np.ndarray = np.array([-.5, .5, 1.5, 2.5])
        probs: np.ndarray = np.ones(values.size) / values.size
        particles: np.ndarray = np.array([-1, 0, 1, 2])

        expected_probs: np.ndarray = np.array(
            [0.5*0.25, 0.25, 0.25, (0.25*1.5)]
        )

        if DEBUG:
            print(f"particles: {particles}")
            print(f"values: {values}")
            print(f"probs: {probs}")

        self.assertTrue(np.isclose(np.sum(expected_probs), 1),
                        "Expected probs do not sum to 1.")

        new_distr: Tuple[np.ndarray, np.ndarray] = categorical_projection(
            (values, probs), particles)

        self.assertTrue(np.equal(new_distr[0], particles).all(),
                        "Particles changed.")

        self.assertTrue(np.isclose(new_distr[1], expected_probs).all(),
                        "Categorical projection failed.")

        return None


if __name__ == "__main__":
    unittest.main()
