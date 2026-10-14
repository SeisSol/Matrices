#!/usr/bin/env python3

import math
import unittest

import numpy as np

from quad_rules import Quadrature
from quad_rules.JaskowiecSukumarSymmetric import JaskowiecSukumarSymmetric
from quad_rules.OrbitSearchTet import OrbitSearchTet
from quad_rules.ReducedSymmetricTet import ReducedSymmetricTet
from quad_rules.SymmetricTet import MIN_BARYCENTRIC, SymmetricTet, orbit_barycentric
from quad_rules.TabulatedTet import barycentric, is_fully_symmetric, read_rule
from quad_rules.WitherdenVincentTet import WitherdenVincentTet
from quad_rules.WitherdenTet import WitherdenTet
from quad_rules.WorkuHickenZinggTet import WorkuHickenZinggTet
from quad_rules.XiaoGimbutasTet import XiaoGimbutasTet

UNIT_TETRAHEDRON = np.array(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
)


def largest_relative_error(points, weights, degree):
    """
    Largest relative error of a rule for the monomials x^a y^b z^c with
    a + b + c <= degree on the unit tetrahedron.
    """
    unit_points, unit_weights = Quadrature.transform(points, weights, UNIT_TETRAHEDRON)
    x, y, z = unit_points.T
    error = 0.0
    for a in range(degree + 1):
        for b in range(degree + 1 - a):
            for c in range(degree + 1 - a - b):
                exact = (
                    math.factorial(a)
                    * math.factorial(b)
                    * math.factorial(c)
                    / math.factorial(a + b + c + 3)
                )
                value = np.dot(unit_weights, x**a * y**b * z**c)
                error = max(error, abs(value - exact) / exact)
    return error


class tabulated_tet_tester(object):
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


class test_witherden_vincent(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = WitherdenVincentTet()
        cls.degrees = [1, 2, 3, 5, 6, 7, 8, 9, 10]
        cls.symmetric = True

    def test_find_best_rule(self):
        points, weights = self.rule.find_best_rule(4)
        self.assertEqual(points.shape, (14, 3))
        self.assertAlmostEqual(weights.sum(), 4.0 / 3.0, delta=1e-14)


class test_witherden(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = WitherdenTet()
        cls.degrees = [10, 11, 12, 13, 15, 16, 17, 18, 19, 20]
        cls.symmetric = True
        # some rules have points on the boundary
        cls.min_barycentric = -1e-14


class test_xiao_gimbutas(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = XiaoGimbutasTet()
        cls.degrees = list(range(1, 16))


class test_jaskowiec_sukumar_symmetric(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = JaskowiecSukumarSymmetric()
        cls.degrees = list(range(1, 21))
        cls.symmetric = True


class test_reduced_symmetric(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = ReducedSymmetricTet()
        cls.degrees = [12, 14]
        cls.symmetric = True
        cls.min_barycentric = 1e-3

    def test_fewer_points(self):
        others = JaskowiecSukumarSymmetric()
        for degree, (points, weights) in self.rule.data.items():
            self.assertLess(points.shape[0], others.data[degree][0].shape[0])


class test_orbit_search(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = OrbitSearchTet()
        cls.degrees = [10]
        cls.symmetric = True
        cls.min_barycentric = 1e-3

    def test_fewer_points(self):
        others = WitherdenVincentTet()
        for degree, (points, weights) in self.rule.data.items():
            self.assertLess(points.shape[0], others.data[degree][0].shape[0])


class test_worku_hicken_zingg(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = WorkuHickenZinggTet()
        cls.degrees = list(range(1, 22))
        cls.symmetric = True


class test_symmetric_selection(tabulated_tet_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rule = SymmetricTet()
        cls.degrees = [1, 2, 3, 5, 6, 7, 8] + list(range(10, 22))
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
        families = [
            WitherdenVincentTet(),
            WitherdenTet(),
            XiaoGimbutasTet(),
            JaskowiecSukumarSymmetric(),
            ReducedSymmetricTet(),
            OrbitSearchTet(),
            WorkuHickenZinggTet(),
        ]
        admissible = []
        for family in families:
            for degree, (points, weights) in family.data.items():
                if (
                    weights.min() > 0
                    and barycentric(points).min() > MIN_BARYCENTRIC
                    and is_fully_symmetric(points, weights)
                ):
                    admissible.append((degree, points.shape[0]))
        for degree in range(22):
            fewest = min(n for d, n in admissible if d >= degree)
            points, _ = self.rule.find_best_rule(degree)
            self.assertEqual(points.shape[0], fewest, f"degree {degree}")

    def test_orbit_order(self):
        # SeisSol reproduces this order of the points of an orbit
        np.testing.assert_allclose(
            orbit_barycentric("S4", [0.1]),
            [
                [0.1, 0.1, 0.1, 0.7],
                [0.1, 0.1, 0.7, 0.1],
                [0.1, 0.7, 0.1, 0.1],
                [0.7, 0.1, 0.1, 0.1],
            ],
            atol=1e-15,
        )
        points = orbit_barycentric("S12", [0.1, 0.2])
        self.assertEqual(points.shape, (12, 4))
        np.testing.assert_allclose(
            points[:3],
            [[0.1, 0.1, 0.2, 0.6], [0.1, 0.1, 0.6, 0.2], [0.1, 0.2, 0.1, 0.6]],
        )


if __name__ == "__main__":
    unittest.main()
