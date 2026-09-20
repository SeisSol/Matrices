#!/usr/bin/env python3

import unittest
import numpy as np

from seissol_matrices import dg_matrices
from seissol_matrices import dr_matrices
from seissol_matrices import quad_points


class abstract_tester(object):
    def compare(self, a, b):
        self.assertEqual(a.shape, b.shape)
        np.testing.assert_allclose(a, b, rtol=0.0, atol=1.2e-11)

    def test_kDivM_material_index_last(self):
        for dim in range(3):
            matrix = self.generator.kDivM(dim, 2)
            self.assertEqual(matrix.shape, (self.nbf, self.nbf, 4))

    def test_kDivM_constant_material(self):
        for dim in range(3):
            self.compare(
                self.generator.kDivM(dim, 1)[:, :, 0], self.generator.kDivM(dim)
            )

    def test_kDivMT_constant_material(self):
        for dim in range(3):
            self.compare(
                self.generator.kDivMT(dim, 1)[:, :, 0], self.generator.kDivMT(dim)
            )

    def test_rDivM_constant_material(self):
        for side in range(4):
            self.compare(
                self.generator.rDivM(side, 1)[:, :, 0], self.generator.rDivM(side)
            )

    def test_V3mTo2nTWDivM_constant_material(self):
        for a in range(4):
            for b in range(4):
                self.compare(
                    self.dr_generator.V3mTo2nTWDivM(a, b, 1)[:, :, 0],
                    self.dr_generator.V3mTo2nTWDivM(a, b),
                )


def setUpClassFromOrder(cls, order):
    cls.order = order
    cls.nbf = order * (order + 1) * (order + 2) // 6
    cls.generator = dg_matrices.dg_generator(order, 3)
    cls.dr_generator = dr_matrices.dr_generator(order, quad_points.stroud(order + 1))


class test_material_order_2(abstract_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 2)


class test_material_order_3(abstract_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 3)


class test_material_order_4(abstract_tester, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 4)


if __name__ == "__main__":
    unittest.main()
