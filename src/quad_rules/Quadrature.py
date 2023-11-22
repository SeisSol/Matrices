import numpy as np
from . import WitherdenVincentTet
from . import WitherdenVincentTri
from . import Dunavant
from . import JaskowiecSukumar


def quad(nodes, weights, f):
    """
    Compute the integral with a given quadrature rule

    Parameters
    ----------
    nodes : np.array of shape (n, d)
        The nodes of the quadrature rule
    weights : np.array of shape (n,)
        The weights of the quadrature rule
    f : function
        The function, of which the integral shall be computed
    """
    f_eval = f(nodes.T)
    return np.dot(f_eval, weights)


def transform(nodes, weights, new_corners):
    """
    Transforms a quadrature rule from the reference element to another element.

    By convention, the reference elements are:
    1D: -1, 1
    2D: (-1, -1), (1, 0), (0, 1)
    3D: (-1, -1, -1), (1, 0, 0), (0, 1, 0), (0, 0, 1)

    Parameters
    ----------
    nodes : np.array of shape (n, d)
        The nodes of the quadrature rule
    weights : np.array of shape (n,)
        The weights of the quadrature rule
    new_corners : np.array of shape (d, d)
        The nodes of the new element
    """
    # 1D nodes are 1-dimensional array
    nodes = nodes.reshape(nodes.shape[0], -1)
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
        x_0 = new_corners[0, :]
        x_1 = new_corners[1, :]
        x_2 = new_corners[2, :]
        x_3 = new_corners[3, :]
        M = np.zeros((3, 3))
        M[:, 0] = 0.5 * (x_1 - x_0)
        M[:, 1] = 0.5 * (x_2 - x_0)
        M[:, 2] = 0.5 * (x_3 - x_0)
        origin = np.array([-1.0, -1.0, -1.0])

    offset = -M @ origin + x_0
    volume_fraction = np.abs(np.linalg.det(M))

    transformed_nodes = np.add(nodes @ M.T, offset)
    transformed_weights = volume_fraction * weights
    return transformed_nodes.reshape(nodes.shape), transformed_weights


def visualize(nodes, weights, new_corners):
    """
    Visualize a quadratur rule on a given element.

    Parameters
    ----------
    nodes : np.array of shape (n, d)
        The nodes of the quadrature rule
    weights : np.array of shape (n,)
        The weights of the quadrature rule
    new_corners : np.array of shape (d, d)
        The nodes of the element
    """
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
    elif nodes.shape[1] == 2:
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
    elif nodes.shape[1] == 3:
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.scatter(nodes[:, 0], nodes[:, 1], nodes[:, 2], s=weights * 40)

        p = new_corners
        for i in range(4):
            verts = [[p[(i) % 4, :], p[(i + 1) % 4, :], p[(i + 2) % 4, :]]]
            srf = Poly3DCollection(verts, alpha=0.25, facecolor="#800000")
            plt.gca().add_collection3d(srf)
        plt.show()


if __name__ == "__main__":
    from scipy.special import comb as choose

    n = 1
    k = 3

    def f(x):
        return x[0] ** n * x[1] ** k

    C = choose(n + k + 1, n)
    print(f"analytic integral = {1 / (k+1) / (n + k + 2) / C}")

    for rule in [WitherdenVincentTri.WitherdenVincentTri(), Dunavant.Dunavant()]:
        nodes, weights = rule.find_best_rule(19)
        new_corners = np.array([[0, 0], [1, 0], [0, 1]])

        nodes_, weights_ = transform(nodes, weights, new_corners)
        visualize(nodes_, weights_, new_corners)
        print(quad(nodes_, weights_, f))
