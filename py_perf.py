import numpy as np


def outer_sum(t1, t2):
    return np.add(t1, t2[:, None]).flatten()


def outer_mul(t1, t2):
    return np.multiply(t1, t2[:, None]).flatten()


def main():
    v1 = np.random.random(100000)
    v1 = v1 / np.sum(v1)
    v2 = np.random.random(100000)
    v2 = v2 / np.sum(v2)

    # v3 = np.add(v1, v2)
    v4 = np.multiply(v1, v2)

    return None


if __name__ == "__main__":
    main()
