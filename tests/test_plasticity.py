#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests
from math import factorial

from seissol_matrices import plasticity_matrices
from seissol_matrices import xml_io
from seissol_matrices import quad_points
from tests import helper


def filename(mode, order):
    return f"plasticity_{mode}_{order}.xml"


class abstract_tester(object):
    def test(self):
        vandermonde = self.generator.generate_Vandermonde(self.mode)
        vandermonde_from_file = xml_io.read_matrix("v", self.filename)
        self.compare_matrices(vandermonde, vandermonde_from_file)

        vandermonde_inv = self.generator.generate_Vandermonde_inv(self.mode)
        vandermonde_inv_from_file = xml_io.read_matrix("vInv", self.filename)
        self.compare_matrices(vandermonde_inv, vandermonde_inv_from_file)


def downlaod_matrix(mode, order):
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/c99ca8f814b7184eb435272f9d4f63b03b8b6cf4/generated_code/matrices/plasticity_{mode}_matrices_{order}.xml"
    r = requests.get(url, allow_redirects=True)
    fn = filename(mode, order)
    open(fn, "wb").write(r.content)


def setUpClassFromOrder(cls, mode, order):
    downlaod_matrix(mode, order)
    cls.filename = filename(mode, order)
    cls.mode = mode
    cls.order = order
    cls.generator = plasticity_matrices.PlasticityGenerator(cls.order)


class test_nb_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "nb", 2)


class test_nb_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "nb", 3)


class test_nb_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "nb", 4)


class test_nb_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "nb", 5)


class test_nb_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "nb", 6)


class test_nb_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "nb", 7)


class test_ip_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "ip", 2)


class test_ip_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "ip", 3)


class test_ip_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "ip", 4)


class test_ip_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "ip", 5)


class test_ip_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "ip", 6)


class test_ip_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, "ip", 7)


class test_sq(unittest.TestCase):
    # number of points of the symmetric rule exact up to degree 2 * (order - 1)
    points = {1: 1, 2: 4, 3: 14, 4: 24, 5: 46, 6: 79, 7: 124, 8: 181}

    def test(self):
        for order, count in self.points.items():
            generator = plasticity_matrices.PlasticityGenerator(order)
            nodes = generator.nodes("sq")
            self.assertEqual(nodes.shape, (count, 3), f"order {order}")
            # all points inside the cell
            barycentric = np.column_stack([1.0 - nodes.sum(axis=1), nodes])
            self.assertGreater(barycentric.min(), 1e-3, f"order {order}")
            # converting to the nodes and back reproduces every basis function
            vandermonde = generator.generate_Vandermonde("sq")
            vandermonde_inv = generator.generate_Vandermonde_inv("sq")
            identity = vandermonde_inv @ vandermonde
            error = np.abs(identity - np.eye(identity.shape[0])).max()
            self.assertLess(error, 1e-12, f"order {order}")


class test_sq3(unittest.TestCase):
    # number of points of the symmetric rule exact up to degree 3 * (order - 1),
    # and up to 2 * order + 1 where that is higher
    points = {1: 8, 2: 14, 3: 35, 4: 59, 5: 124, 6: 213, 7: 352, 8: 540}

    def test(self):
        for order, count in self.points.items():
            generator = plasticity_matrices.PlasticityGenerator(order)
            nodes = generator.nodes("sq3")
            self.assertEqual(nodes.shape, (count, 3), f"order {order}")
            barycentric = np.column_stack([1.0 - nodes.sum(axis=1), nodes])
            self.assertGreater(barycentric.min(), 1e-3, f"order {order}")
            vandermonde = generator.generate_Vandermonde("sq3")
            vandermonde_inv = generator.generate_Vandermonde_inv("sq3")
            identity = vandermonde_inv @ vandermonde
            error = np.abs(identity - np.eye(identity.shape[0])).max()
            self.assertLess(error, 1e-12, f"order {order}")

    def test_product_of_three_basis_functions(self):
        # every monomial up to the degree of such a product integrates exactly;
        # over the unit tetrahedron, x^a y^b z^c gives a! b! c! / (a + b + c + 3)!
        for order in self.points:
            nodes, weights = plasticity_matrices.PlasticityGenerator(
                order
            ).quadrature_rule("sq3")
            degree = max(3 * (order - 1), 2 * order + 1)
            for a in range(degree + 1):
                for b in range(degree + 1 - a):
                    for c in range(degree + 1 - a - b):
                        exact = (
                            factorial(a)
                            * factorial(b)
                            * factorial(c)
                            / factorial(a + b + c + 3)
                        )
                        value = weights @ (
                            nodes[:, 0] ** a * nodes[:, 1] ** b * nodes[:, 2] ** c
                        )
                        self.assertLess(
                            abs(value / exact - 1.0),
                            1e-10,
                            f"order {order} {a} {b} {c}",
                        )


if __name__ == "__main__":
    unittest.main()
