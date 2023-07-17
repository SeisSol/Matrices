import numpy as np

from quad_rules.QuadRule import QuadRule


class GaussJacobi(QuadRule):
    def __init__(self, alpha, beta):
        self.a = alpha
        self.b = beta
        self.eps = 1e-14

    def compute_nodes_and_weights(self, n):
        weightFactor = (
            -(2.0 * n + self.a + self.b + 2)
            * np.math.factorial(n + self.a)
            * np.math.factorial(n + self.b)
            * 2 ** (self.a + self.b)
            / (
                (n + self.a + self.b + 1.0)
                * np.math.factorial(n + self.a + self.b)
                * np.math.factorial(n + 1)
            )
        )
        nodes = np.zeros((n, 1))
        weights = np.zeros(n)
        for i in range(1, n + 1):
            x = np.cos(
                np.pi * (0.5 * self.a + i - 0.25) / (0.5 * (1.0 + self.a + self.b) + n)
            )
            p = jacobi_polynomial(n, self.a, self.b, x)
            d_p = jacobi_derivative(n, self.a, self.b, x)
            while np.abs(p) > self.eps:
                x -= p / d_p
                p = jacobi_polynomial(n, self.a, self.b, x)
                d_p = jacobi_derivative(n, self.a, self.b, x)
            nodes[i - 1, 0] = x
            weights[i - 1] = weightFactor / (
                jacobi_polynomial(n + 1, self.a, self.b, x) * d_p
            )
        return nodes, weights


def jacobi_polynomial(n, a, b, x):
    if n == 0:
        return 1.0
    p_1 = 1.0
    p = 0.5 * a - 0.5 * b + (1.0 + 0.5 * (a + b)) * x
    a2_b2 = a * a - b * b
    for m in range(2, n + 1):
        p_2 = p_1
        p_1 = p
        p = (
            (2.0 * m + a + b - 1.0)
            * (a2_b2 + (2.0 * m + a + b) * (2.0 * m + a + b - 2.0) * x)
            * p_1
            - 2.0 * (m + a - 1.0) * (m + b - 1.0) * (2.0 * m + a + b) * p_2
        ) / (2.0 * m * (m + a + b) * (2.0 * m + a + b - 2.0))
    return p


def jacobi_derivative(n, a, b, x):
    if n == 0:
        return 0.0
    else:
        return 0.5 * (n + a + b + 1.0) * jacobi_polynomial(n - 1, a + 1, b + 1, x)


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    x = np.linspace(-1, 1, 101)
    n = 5
    y = [jacobi_polynomial(n, 0, 0, s) for s in x]
    dy = [jacobi_derivative(n, 0, 0, s) for s in x]
    nodes, weights = GaussJacobi(0, 0).compute_nodes_and_weights(n)
    print(weights)
    plt.plot(x, y)
    # plt.plot(x, dy)
    plt.scatter(nodes, np.zeros(nodes.shape))
    plt.show()
