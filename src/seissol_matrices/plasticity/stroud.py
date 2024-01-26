import sys

sys.path.append("../..")
from quad_rules import GaussJacobi
import numpy as np

np.set_printoptions(precision=15, floatmode="fixed")


def stroud(n):
    points_0, weights_0 = GaussJacobi.GaussJacobi(0, 0).compute_nodes_and_weights(n)
    points_1, weights_1 = GaussJacobi.GaussJacobi(1, 0).compute_nodes_and_weights(n)
    points_2, weights_2 = GaussJacobi.GaussJacobi(2, 0).compute_nodes_and_weights(n)

    x = np.zeros((n**3, 3))
    w = np.zeros(n**3)
    for i, index in enumerate(np.ndindex(n, n, n)):
        x[i, :] = np.array([points_2[index[0]], points_1[index[1]], points_0[index[2]]])
        w[i] = (
            weights_2[index[0]] * weights_1[index[1]] * weights_0[index[2]] * 0.5**6
        )
    x = 0.5 * x + 0.5

    y = np.zeros((n**3, 3))
    y[:, 0] = x[:, 0]
    y[:, 1] = x[:, 1] * (1 - y[:, 0])
    y[:, 2] = x[:, 2] * (1 - y[:, 0] - y[:, 1])

    return y, w


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("n", type=int)
    args = parser.parse_args()

    y, w = stroud(args.n)
    for p in y:
        print(f"{p[0]} {p[1]} {p[2]}")
