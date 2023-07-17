import numpy as np
from . import WitherdenVincentTet
from . import WitherdenVincentTri
from . import GaussJacobi


def quad(nodes, weights, f):
    f_eval = f(nodes.T)
    return np.dot(f_eval, weights)


def transform(nodes, weights, new_corners):
    if nodes.shape[1] == 1:
        x_0 = new_corners[0, :]
        x_1 = new_corners[1, :]
        M = np.zeros((1, 1))
        M[:, 0] = 0.5 * (x_1 - x_0)
        origin = np.array([-1.0])
    elif nodes.shape[1] == 2:
        x_0 = new_corners[0, :]
        x_1 = new_corners[1, :]
        x_2 = new_corners[2, :]
        M = np.zeros((2, 2))
        M[:, 0] = 0.5 * (x_1 - x_0)
        M[:, 1] = 0.5 * (x_2 - x_0)
        origin = np.array([-1.0, -1.0])
    elif nodes.shape[1] == 3:
        x_0 = nodes[0, :]
        x_1 = nodes[1, :]
        x_2 = nodes[2, :]
        x_3 = nodes[3, :]
        M = np.zeros((3, 3))
        M[:, 0] = 0.5 * (x_1 - x_0)
        M[:, 1] = 0.5 * (x_2 - x_0)
        M[:, 2] = 0.5 * (x_3 - x_0)
        origin = np.array([-1.0, -1.0, -1.0])

    offset = -M @ origin + x_0
    volume_fraction = np.abs(np.linalg.det(M))
    return np.add(nodes @ M.T, offset), volume_fraction * weights


def visualize(nodes, weights, new_corners):
    import matplotlib.pyplot as plt

    if nodes.shape[1] == 1:
        plt.scatter(nodes[:], np.zeros(nodes.shape), s=weights * 40)
        plt.plot(
            [
                new_corners[0, 0],
                new_corners[0, 0],
                new_corners[1, 0],
                new_corners[1, 0],
                new_corners[0, 0],
            ],
            [0.5, -0.5, -0.5, 0.5, 0.5],
        )
        plt.show()
    if nodes.shape[1] == 2:
        plt.scatter(nodes[:, 0], nodes[:, 1], s=weights * 40)
        plt.plot(
            [
                new_corners[0, 0],
                new_corners[1, 0],
                new_corners[2, 0],
                new_corners[0, 0],
            ],
            [
                new_corners[0, 1],
                new_corners[1, 1],
                new_corners[2, 1],
                new_corners[0, 1],
            ],
        )
        plt.show()


if __name__ == "__main__":
    # nodes, weights = WitherdenVincentTri.WitherdenVincentTri().find_best_rule(3)
    # new_corners = np.array([[-2, -1], [4, -3], [-1, 7]])
    # nodes, weights = WitherdenVincentTet.WitherdenVincentTet().find_best_rule(3)
    nodes, weights = GaussJacobi.GaussJacobi(0, 0).find_best_rule(6)
    new_corners = np.array([[-2], [4]])

    def f(x):
        return 1

    nodes_, weights_ = transform(nodes, weights, new_corners)
    visualize(nodes_, weights_, new_corners)
