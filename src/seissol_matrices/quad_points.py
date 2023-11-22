from abc import ABC
import numpy as np
import quad_rules.Quadrature
import quad_rules.GaussJacobi
import quad_rules.Dunavant
import quad_rules.WitherdenVincentTri


class quadrature(ABC):
    def points(self):
        pass

    def weights(self):
        pass

    def id(self):
        return f"{self.name}: {self.n}"


class stroud(quadrature):
    def __init__(self, n):
        self.n = n
        self.name = "Stroud"

    def gauss_jacobi_quadrature_1d(self, a, b):
        nodes, weights = quad_rules.GaussJacobi.GaussJacobi(a, b).find_best_rule(self.n)
        return nodes, weights

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
        weights = weights.reshape(((self.n) ** 2))
        return weights


class dunavant(quadrature):
    def __init__(self, n):
        self.n = n
        self.name = "Dunavant"

    def points(self):
        n, w = quad_rules.Dunavant.Dunavant().find_best_rule(self.n)
        n_, w_ = quad_rules.Quadrature.transform(
            n, w, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        return n_

    def weights(self):
        n, w = quad_rules.Dunavant.Dunavant().find_best_rule(self.n)
        n_, w_ = quad_rules.Quadrature.transform(
            n, w, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        return w_


class witherden_vincent(quadrature):
    def __init__(self, n):
        # Witherden Vincent Quadratur rule of order 4 does not exist
        self.n = n if n != 3 else 4
        self.name = "Witherden Vincent"

    def points(self):
        n, w = quad_rules.WitherdenVincentTri.WitherdenVincentTri().find_best_rule(
            self.n
        )
        n_, w_ = quad_rules.Quadrature.transform(
            n, w, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        return n_

    def weights(self):
        n, w = quad_rules.WitherdenVincentTri.WitherdenVincentTri().find_best_rule(
            self.n
        )
        n_, w_ = quad_rules.Quadrature.transform(
            n, w, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )
        return w_


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    reference_rule = witherden_vincent(19)
    N = 6
    p = 4
    polynomial = lambda x: x[0] ** p * x[1] ** (N - p)
    reference_points = reference_rule.points()
    reference_weights = reference_rule.weights()
    reference_integral = np.dot(polynomial(reference_points.T), reference_weights)
    print(reference_integral)

    def find_required_degree(rule):
        d = 0
        while True:
            r = rule(d)
            points = r.points()
            weights = r.weights()
            integral = np.dot(polynomial(points.T), weights)
            if np.abs(reference_integral - integral) < 1e-15:
                return d
            else:
                d += 1

    dunavant_d = find_required_degree(dunavant)
    print("Dunavant", dunavant_d)

    stroud_d = find_required_degree(stroud)
    print("Stroud", stroud_d)

    wv_d = find_required_degree(witherden_vincent)
    print("Witherden Vincent", wv_d)

    tri_x = [0, 1, 0, 0]
    tri_y = [0, 0, 1, 0]
    plt.plot(tri_x, tri_y, color="darkgrey")
    for quadrule in [dunavant(dunavant_d), stroud(stroud_d), witherden_vincent(wv_d)]:
        quadpoints = quadrule.points()
        quadweights = quadrule.weights()
        plt.scatter(
            quadpoints[:, 0], quadpoints[:, 1], s=200 * quadweights, label=quadrule.id()
        )
    plt.legend()
    plt.show()
