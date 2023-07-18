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
        ax = fig.add_subplot(projection='3d')
        ax.scatter(nodes[:, 0], nodes[:, 1],nodes[:, 2],  s=weights * 40)

        p = new_corners
        for i in range(4):
            verts =  [[p[(i) % 4,:], p[(i+1) % 4,:], p[(i+2) % 4,:]]]
            srf = Poly3DCollection(verts, alpha=.25, facecolor='#800000')
            plt.gca().add_collection3d(srf)
        plt.show()


if __name__ == "__main__":
    # nodes, weights = WitherdenVincentTri.WitherdenVincentTri().find_best_rule(3)
    # new_corners = np.array([[-2, -1], [4, -3], [-1, 7]])
    nodes, weights = WitherdenVincentTet.WitherdenVincentTet().find_best_rule(7)
    new_corners = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    # nodes, weights = WitherdenVincentTet.WitherdenVincentTet().find_best_rule(3)
    # nodes, weights = GaussJacobi.GaussJacobi(0, 0).find_best_rule(6)
    # new_corners = np.array([[-2], [4]])

    def f(x):
        #return np.ones(x[0].shape)
        return x[0] * x[1]**2 * x[2]**3

    nodes_, weights_ = transform(nodes, weights, new_corners)
    visualize(nodes_, weights_, new_corners)
    print(quad(nodes_, weights_, f))
