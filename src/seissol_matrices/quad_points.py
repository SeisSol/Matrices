from abc import ABC
import numpy as np
import quad_rules.Quadrature
import quad_rules.GaussJacobi
import quad_rules.Dunavant
import quad_rules.WitherdenVincentTri
import quad_rules.SymmetricTri


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
        self.name = "stroud"

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


def interior_rule(family, degree):
    """
    The rule of a family with the fewest points that is exact up to at least the
    given degree, has positive weights and all points inside the triangle.

    A rule with a point outside or on the boundary of a face would sample a fault
    there, or beyond it.
    """
    candidates = []
    for d, (nodes, weights) in family.data.items():
        unit = 0.5 * (nodes + 1.0)
        smallest = min(unit.min(), (1.0 - unit.sum(axis=1)).min())
        if d >= degree and weights.min() > 0 and smallest > 0:
            candidates.append((len(weights), d))
    if not candidates:
        raise ValueError(f"no interior rule with positive weights of degree {degree}")
    _, best = min(candidates)
    return family.data[best]


class triangle_family_rule(quadrature):
    """
    A rule of a family of triangular rules, exact up to at least the given degree,
    with positive weights and all points inside, on the unit triangle.
    """

    def __init__(self, degree, family, name):
        self.n = degree
        self.name = name
        nodes, weights = interior_rule(family, degree)
        self.nodes, self.weights_ = quad_rules.Quadrature.transform(
            nodes, weights, np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
        )

    def points(self):
        return self.nodes

    def weights(self):
        return self.weights_


class dunavant(triangle_family_rule):
    def __init__(self, degree):
        super().__init__(degree, quad_rules.Dunavant.Dunavant(), "dunavant")


class witherden_vincent(triangle_family_rule):
    def __init__(self, degree):
        super().__init__(
            degree,
            quad_rules.WitherdenVincentTri.WitherdenVincentTri(),
            "witherden_vincent",
        )


class symmetric(triangle_family_rule):
    def __init__(self, degree):
        super().__init__(degree, quad_rules.SymmetricTri.SymmetricTri(), "symmetric")


def dr_rules(order):
    """
    The rules a fault face is sampled with, by name, for a scheme of the given
    order (whose basis has the degree order - 1): Stroud's conical product with
    order + 1 points per direction, exact up to degree 2 * order + 1, and the
    Dunavant and Witherden-Vincent rules exact up to degree 2 * order, two
    degrees more than the mass matrix of a face needs.
    """
    return {
        "stroud": stroud(order + 1),
        "dunavant": dunavant(2 * order),
        "witherden_vincent": witherden_vincent(2 * order),
    }


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
