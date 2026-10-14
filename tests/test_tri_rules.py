#!/usr/bin/env python3

import math
import unittest

import numpy as np

from quad_rules import Quadrature
from quad_rules.SymmetricTri import MIN_BARYCENTRIC, SymmetricTri, orbit_barycentric
from quad_rules.TabulatedTri import barycentric, is_fully_symmetric, read_rule
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


class test_symmetric_selection(tabulated_tri_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = SymmetricTri()
        cls.degrees = [1, 2] + list(range(4, 21))
        cls.symmetric = True
        cls.min_barycentric = MIN_BARYCENTRIC

    def test_same_rules_as_files(self):
        # reordered by orbits, the rules hold the points and weights of their files
        for degree, path in self.rule.files.items():
            points, weights = read_rule(path)
            ordered, ordered_weights = self.rule.data[degree]
            distance = np.linalg.norm(points[:, None, :] - ordered[None, :, :], axis=2)
            partner = np.argmin(distance, axis=1)
            self.assertLess(
                distance[np.arange(len(points)), partner].max(),
                1e-13,
                f"degree {degree}",
            )
            self.assertEqual(
                sorted(partner), list(range(len(points))), f"degree {degree}"
            )
            self.assertLess(
                np.abs(weights - ordered_weights[partner]).max(),
                1e-14,
                f"degree {degree}",
            )

    def test_fewest_points(self):
        # the fewest points, and of those (up to 1e-6) the largest smallest
        # barycentric coordinate
        admissible = []
        for family in [WitherdenVincentTri(), XiaoGimbutasTri()]:
            for degree, (points, weights) in family.data.items():
                smallest = barycentric(points).min()
                if (
                    weights.min() > 0
                    and smallest > MIN_BARYCENTRIC
                    and is_fully_symmetric(points, weights)
                ):
                    admissible.append((degree, points.shape[0], smallest))
        for degree in range(21):
            fewest = min(n for d, n, _ in admissible if d >= degree)
            inside = max(b for d, n, b in admissible if d >= degree and n == fewest)
            points, _ = self.rule.find_best_rule(degree)
            self.assertEqual(points.shape[0], fewest, f"degree {degree}")
            self.assertGreater(
                barycentric(points).min(), inside - 1e-6, f"degree {degree}"
            )

    def test_orbit_order(self):
        np.testing.assert_allclose(
            orbit_barycentric("S3", [0.1]),
            [[0.1, 0.1, 0.8], [0.1, 0.8, 0.1], [0.8, 0.1, 0.1]],
            atol=1e-15,
        )
        np.testing.assert_allclose(
            orbit_barycentric("S6", [0.1, 0.2]),
            [
                [0.1, 0.2, 0.7],
                [0.1, 0.7, 0.2],
                [0.2, 0.1, 0.7],
                [0.2, 0.7, 0.1],
                [0.7, 0.1, 0.2],
                [0.7, 0.2, 0.1],
            ],
            atol=1e-15,
        )

    def test_vertex_permutations_permute_points(self):
        # the point sets of the two cells sharing a face coincide
        for degree, (points, weights) in self.rule.data.items():
            lam = barycentric(points)
            for permutation in [(1, 0, 2), (0, 2, 1), (2, 1, 0), (1, 2, 0)]:
                moved = lam[:, list(permutation)]
                distance = np.linalg.norm(moved[:, None, :] - lam[None, :, :], axis=2)
                partner = np.argmin(distance, axis=1)
                self.assertLess(distance.min(axis=1).max(), 1e-14, f"degree {degree}")
                self.assertEqual(
                    sorted(partner), list(range(len(lam))), f"degree {degree}"
                )


if __name__ == "__main__":
    unittest.main()
