#!/usr/bin/env python3

import math
import unittest

import numpy as np

from quad_rules import Quadrature
from quad_rules.TabulatedTri import barycentric, is_fully_symmetric
from quad_rules.WitherdenVincentTri import WitherdenVincentTri
from quad_rules.XiaoGimbutasTri import XiaoGimbutasTri

UNIT_TRIANGLE = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])


def largest_relative_error(points, weights, degree):
    """
    Largest relative error of a rule for the monomials x^a y^b with
    a + b <= degree on the unit triangle.
    """
    unit_points, unit_weights = Quadrature.transform(points, weights, UNIT_TRIANGLE)
    x, y = unit_points.T
    error = 0.0
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            exact = math.factorial(a) * math.factorial(b) / math.factorial(a + b + 2)
            value = np.dot(unit_weights, x**a * y**b)
            error = max(error, abs(value - exact) / exact)
    return error


class tabulated_tri_tester(object):
    # all rules of the family are invariant under permutations of the vertices
    symmetric = False
    # smallest barycentric coordinate allowed for the points
    min_barycentric = 0.0

    def test_degrees(self):
        self.assertEqual(sorted(self.rule.data.keys()), self.degrees)

    def test_exact(self):
        for degree, (points, weights) in self.rule.data.items():
            error = largest_relative_error(points, weights, degree)
            self.assertLess(error, 1e-12, f"degree {degree}")

    def test_positive(self):
        for degree, (points, weights) in self.rule.data.items():
            self.assertGreater(weights.min(), 0.0, f"degree {degree}")

    def test_inside(self):
        for degree, (points, weights) in self.rule.data.items():
            smallest = barycentric(points).min()
            self.assertGreater(smallest, self.min_barycentric, f"degree {degree}")

    def test_symmetric(self):
        if not self.symmetric:
            self.skipTest("the family is not fully symmetric")
        for degree, (points, weights) in self.rule.data.items():
            self.assertTrue(is_fully_symmetric(points, weights), f"degree {degree}")


class test_witherden_vincent(tabulated_tri_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = WitherdenVincentTri()
        cls.degrees = [1, 2] + list(range(4, 21))
        cls.symmetric = True
        # the rule of degree 16 has points on the boundary
        cls.min_barycentric = -1e-14

    def test_find_best_rule(self):
        points, weights = self.rule.find_best_rule(3)
        self.assertEqual(points.shape, (6, 2))
        self.assertAlmostEqual(weights.sum(), 2.0, delta=1e-14)


class test_xiao_gimbutas(tabulated_tri_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = XiaoGimbutasTri()
        cls.degrees = list(range(1, 31))
        cls.symmetric = True


if __name__ == "__main__":
    unittest.main()
