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

    def test_rDivM_material_on_the_face(self):
        # M^-1 (integral over the face of the cell basis times the face basis
        # times the material basis) M_face^-1, by a quadrature of its own
        generator = self.generator
        material = dg_matrices.dg_generator(3, 3).generator
        nodes, weights = generator.face_generator.get_quadrature_rule(
            2 * self.order + 4
        )
        points = nodes.T
        face_basis = generator.face_generator.generator.eval_all(points)
        Mf = generator.face_generator.mass_matrix()
        M = generator.mass_matrix()
        for side in range(4):
            volume_points = generator.volume_to_face_parametrisation(points, side)
            volume = generator.generator.eval_all(volume_points)
            materials = material.eval_all(volume_points)
            matrix = generator.rDivM(side, 3)
            for m in range(materials.shape[1]):
                integral = np.einsum(
                    "q,qk,qj->kj", weights * materials[:, m], volume, face_basis
                )
                expected = np.linalg.solve(M, np.linalg.solve(Mf, integral.T).T)
                self.compare(matrix[:, :, m], expected)

    def test_rDivM_material_vanishing_on_the_face(self):
        # a material that vanishes on a face, the barycentric coordinate of the
        # vertex opposite to it, gives nothing through that face
        material = dg_matrices.dg_generator(2, 3)
        nodes, weights = material.get_quadrature_rule(4)
        points = nodes.T
        values = material.generator.eval_all(points)
        for side, vertex in enumerate((3, 2, 1, 0)):
            barycentric = (
                1.0 - points.sum(axis=0) if vertex == 0 else points[vertex - 1]
            )
            coefficients = np.linalg.solve(
                material.mass_matrix(), values.T @ (weights * barycentric)
            )
            lift = self.generator.rDivM(side, 2) @ coefficients
            np.testing.assert_allclose(lift, 0.0, rtol=0.0, atol=1e-11)
            self.assertGreater(
                np.abs(self.generator.rDivM((side + 1) % 4, 2) @ coefficients).max(),
                1e-3,
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
