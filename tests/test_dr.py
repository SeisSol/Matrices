#!/usr/bin/env python3

import math
import unittest
import numpy as np
import os
import requests

from seissol_matrices import dr_matrices
from seissol_matrices import json_io
from seissol_matrices import quad_points
from tests import helper


class abstract_tester(object):
    def test_V3mTo2n(self):
        for a in range(0, 4):
            for b in range(0, 4):
                V3mTo2n = self.generator.V3mTo2n(a, b)
                V3mTo2n_from_file = json_io.read_matrix(
                    f"V3mTo2n({a},{b})", self.filename
                )
                self.compare_matrices(V3mTo2n, V3mTo2n_from_file)

    def test_V3mTo2nTWDivM(self):
        np.set_printoptions(linewidth=200)
        for a in range(4):
            for b in range(4):
                V3mTo2nTWDivM = self.generator.V3mTo2nTWDivM(a, b)
                V3mTo2nTWDivM_from_file = json_io.read_matrix(
                    f"V3mTo2nTWDivM({a},{b})", self.filename
                )
                self.compare_matrices(V3mTo2nTWDivM, V3mTo2nTWDivM_from_file)


def setUpClassFromOrder(cls, order):
    cls.filename = f"dr_quadrature_matrices_{order}.json"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/c99ca8f814b7184eb435272f9d4f63b03b8b6cf4/generated_code/matrices/dr_quadrature_matrices_{order}.json"
    r = requests.get(url, allow_redirects=True)
    with open(cls.filename, "wb") as f:
        f.write(r.content)
    cls.order = order
    quadrule = quad_points.stroud(order + 1)
    cls.generator = dr_matrices.dr_generator(cls.order, quadrule)


class test_dr_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 2)


class test_dr_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 3)


class test_dr_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 4)


class test_dr_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 5)


class test_dr_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 6)


class test_dr_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 7)


class test_dr_rules(unittest.TestCase):
    def test_exact_positive_inside(self):
        for order in range(2, 9):
            for name, rule in quad_points.dr_rules(order).items():
                degree = 2 * order + 1 if name == "stroud" else 2 * order
                x, y = rule.points().T
                weights = rule.weights()
                for a in range(degree + 1):
                    for b in range(degree + 1 - a):
                        exact = (
                            math.factorial(a)
                            * math.factorial(b)
                            / math.factorial(a + b + 2)
                        )
                        value = np.dot(weights, x**a * y**b)
                        self.assertLess(
                            abs(value - exact) / exact, 1e-12, f"{name} {order}"
                        )
                self.assertGreater(weights.min(), 0.0, f"{name} {order}")
                smallest = min(x.min(), y.min(), (1.0 - x - y).min())
                self.assertGreater(smallest, 0.0, f"{name} {order}")


# the dynamic rupture matrices SeisSol uses with the Dunavant rule
SEISSOL_COMMIT = "1a3d2d53aa4618af4b1c2405ee40f63cfb191f99"


class dunavant_tester(object):
    def test_same_rule(self):
        # the same points and weights as the files, in another order
        points = json_io.read_matrix("quadpoints", self.filename)
        weights = json_io.read_matrix("quadweights", self.filename).flatten()
        self.assertEqual(points.shape, self.rule.points().shape)
        np.testing.assert_allclose(
            self.rule.points()[self.order_of_file], points, atol=1e-15
        )
        np.testing.assert_allclose(
            self.rule.weights()[self.order_of_file], weights, atol=1e-15
        )

    def test_V3mTo2nTWDivM(self):
        for a in range(4):
            for b in range(4):
                V3mTo2nTWDivM = self.generator.V3mTo2nTWDivM(a, b)
                from_file = json_io.read_matrix(
                    f"V3mTo2nTWDivM({a},{b})", self.filename
                )
                self.compare_matrices(V3mTo2nTWDivM[:, self.order_of_file], from_file)


def setUpDunavantFromOrder(cls, order):
    cls.filename = f"dr_dunavant_matrices_{order}.json"
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/{SEISSOL_COMMIT}/codegen/matrices/dr_dunavant_matrices_{order}.json"
    r = requests.get(url, allow_redirects=True)
    with open(cls.filename, "wb") as f:
        f.write(r.content)
    cls.rule = quad_points.dr_rules(order)["dunavant"]
    cls.generator = dr_matrices.dr_generator(order, cls.rule)
    points = json_io.read_matrix("quadpoints", cls.filename)
    distance = np.linalg.norm(
        points[:, None, :] - cls.rule.points()[None, :, :], axis=2
    )
    cls.order_of_file = np.argmin(distance, axis=1)


class test_dr_dunavant_2(dunavant_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpDunavantFromOrder(cls, 2)


class test_dr_dunavant_4(dunavant_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpDunavantFromOrder(cls, 4)


class test_dr_dunavant_7(dunavant_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpDunavantFromOrder(cls, 7)


if __name__ == "__main__":
    unittest.main()
