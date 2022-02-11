from abc import ABC
import quadpy as qp
import numpy as np


class quadrature(ABC):
    def points(self):
        pass

    def weights(self):
        pass

    def id(self):
        return f"{self.name}: {self.n}"


class gauss_jacobi(quadrature):
    def __init__(self, n):
        self.n = n
        self.name = "jacobi"

    def gauss_jacobi_quadrature_1d(self, a, b):
        gauss_jacobi_rule = qp.c1.gauss_jacobi(self.n, a, b)
        return gauss_jacobi_rule.points[::-1], gauss_jacobi_rule.weights[::-1]

    def points(self):
        points0, weights0 = self.gauss_jacobi_quadrature_1d(0, 0)
        points1, weights1 = self.gauss_jacobi_quadrature_1d(1, 0)
        points = np.zeros((self.n, self.n, 2))
        for i in range(self.n):
            for j in range(self.n):
                points[i, j, 0] = 0.5 * (1.0 + points1[i])
                points[i, j, 1] = 0.25 * (1.0 + points0[j]) * (1.0 - points1[i])
        points = points.reshape(((self.n) ** 2, 2))
        return points

    def weights(self):
        points0, weights0 = self.gauss_jacobi_quadrature_1d(0, 0)
        points1, weights1 = self.gauss_jacobi_quadrature_1d(1, 0)
        weights = np.zeros((self.n, self.n))
        for i in range(self.n):
            for j in range(self.n):
                weights[i, j] = weights1[i] * weights0[j] * 0.125
        weights = weights.reshape(((self.n) ** 2, 1))
        return weights


class dunavant(quadrature):
    def __init__(self, n):
        self.n = n
        self.name = "dunavant"

    def points(self):
        dunavant = qp.t2.schemes[f"dunavant_{self.n:02d}"]()
        return dunavant.points[:2, :].T

    def weights(self):
        dunavant = qp.t2.schemes[f"dunavant_{self.n:02d}"]()
        return 0.5 * np.reshape(dunavant.weights, (dunavant.weights.shape[0], 1))


class witherden_vincent(quadrature):
    def __init__(self, n):
        # Witherden Vincent Quadratur rule of order 4 does not exist
        self.n = n if n != 3 else 4
        self.name = "witherden_vincent"

    def points(self):
        wv = qp.t2.schemes[f"witherden_vincent_{self.n:02d}"]()
        return wv.points[:2, :].T

    def weights(self):
        wv = qp.t2.schemes[f"witherden_vincent_{self.n:02d}"]()
        return 0.5 * np.reshape(wv.weights, (wv.weights.shape[0], 1))


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    o = 6
    for quadrule in [witherden_vincent(o + 1), dunavant(o + 1), gauss_jacobi(o + 1)]:
        tri_x = [0, 1, 0, 0]
        tri_y = [0, 0, 1, 0]
        plt.plot(tri_x, tri_y)
        quadpoints = quadrule.points()
        plt.scatter(quadpoints[:, 0], quadpoints[:, 1])
        plt.title(quadrule.id())
        plt.show()
