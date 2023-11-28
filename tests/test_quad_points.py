#!/usr/bin/env python3

from abc import ABC
import unittest
import numpy as np
import sympy

from seissol_matrices import quad_points


def test_function(x, y, d_1, d_2):
    return x**d_1 * y**d_2


def analytical_result(d_1, d_2):
    x = sympy.Symbol("x")
    y = sympy.Symbol("y")
    p = x**d_1 * y**d_2
    int_in_y = sympy.integrate(p, (y, 0, 1 - x))
    return sympy.integrate(int_in_y, (x, 0, 1)).evalf()


class base(ABC):
    def test(self):
        nodes = self.rule.points()
        x = nodes[:, 0]
        y = nodes[:, 1]
        weights = self.rule.weights()
        for d_1 in range(3):
            for d_2 in range(3):
                f = test_function(x, y, d_1, d_2)
                integral = np.dot(f, weights)
                reference = analytical_result(d_1, d_2)
                self.assertAlmostEqual(
                    reference,
                    integral,
                    None,
                    f"Integral not correct x**{d_1} * x**{d_2}",
                    1e-12,
                )


class test_stroud(base, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_stroud, self).__init__(*args, **kwargs)
        self.rule = quad_points.stroud(5)


class test_dunavant(base, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_dunavant, self).__init__(*args, **kwargs)
        self.rule = quad_points.dunavant(5)


class test_witherden_vincent(base, unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(test_witherden_vincent, self).__init__(*args, **kwargs)
        self.rule = quad_points.witherden_vincent(5)


if __name__ == "__main__":
    unittest.main()
