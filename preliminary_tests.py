"""Preliminary tests for master thesis."""

# import numpy as np


import scipy.stats as sp


STATES = {1, 2}
ACTIONS = {1, 2, 3, 4}

# need some initial collection of distrs for the total reward =: nu^0
# need some return distributons
# need samples from return distr for each triple (s, a, s') (N * |S| * |A| * |S|) many)
# need Bellman operator
# need some policy (fixed), define state - action dynamics
# need some distributions such that everything is known and can be compared
# define random bellman operator as partial function

# how does convuliton look like if one of the distrs is discrete, empirical (or both)


def main():
    """Call main function."""
    # rt = sp.norm(loc=0, scale=1)
    bernoulli_scaled = sp.rv_discrete(values=([-1, 1], [0.5, 0.5]))
    print(bernoulli_scaled.rvs(size=10))
    
if __name__ == "__main__":
    main()
