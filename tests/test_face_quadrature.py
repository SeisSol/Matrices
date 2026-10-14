#!/usr/bin/env python3

import itertools
import unittest

import numpy as np

from seissol_matrices import (
    basis_functions,
    dg_matrices,
    face_quadrature,
    nodal_boundary,
)


def face_to_cell(face, point):
    """A point of the reference triangle on a face of the reference tetrahedron, as SeisSol
    parametrises the faces of a cell."""
    x, y = point
    return {
        0: [y, x, 0.0],
        1: [x, 0.0, y],
        2: [0.0, y, x],
        3: [1.0 - x - y, x, y],
    }[face]


class test_face_quadrature(unittest.TestCase):
    def test_points(self):
        for order in range(2, 9):
            generator = face_quadrature.FaceQuadratureGenerator(order)
            self.assertAlmostEqual(generator.weights.sum(), 0.5, delta=1e-15)
            self.assertGreater(generator.weights.min(), 0.0)
            x, y = generator.points.T
            self.assertGreater(min(x.min(), y.min(), (1 - x - y).min()), 1e-3)

    def test_mass_matrix(self):
        # the rule integrates the products of two face basis functions exactly
        for order in range(2, 9):
            generator = face_quadrature.FaceQuadratureGenerator(order)
            mass = dg_matrices.dg_generator(order, d=2).mass_matrix()
            product = (
                generator.generate_VFaceQuadTo2m() @ generator.generate_V2mToFaceQuad()
            )
            np.testing.assert_allclose(product, mass, atol=1e-14)

    def test_V3mToFaceQuad(self):
        # the volume basis functions at the points of each face
        for order in range(2, 9):
            generator = face_quadrature.FaceQuadratureGenerator(order)
            basis = basis_functions.BasisFunctionGenerator3D(order)
            count = basis.number_of_basis_functions()
            for face in range(4):
                direct = np.array(
                    [
                        [
                            basis.eval_basis(face_to_cell(face, p), j)
                            for j in range(count)
                        ]
                        for p in generator.points
                    ]
                )
                np.testing.assert_allclose(
                    generator.generate_V3mToFaceQuad(face), direct, atol=1e-13
                )

    def test_moments_as_nodal(self):
        # for a polynomial on the face, the same moments as the nodal set gives
        for order in range(2, 8):
            generator = face_quadrature.FaceQuadratureGenerator(order)
            nodal = nodal_boundary.NodalBoundaryGenerator(order)
            coefficients = np.random.default_rng(order).standard_normal(
                nodal.vandermonde.shape[1]
            )
            quadrature = generator.generate_VFaceQuadTo2m() @ (
                generator.generate_V2mToFaceQuad() @ coefficients
            )
            interpolation = nodal.generate_V2nTo2m() @ (
                nodal.generate_V2mTo2n() @ coefficients
            )
            np.testing.assert_allclose(quadrature, interpolation, atol=1e-13)

    def test_vertex_permutations(self):
        # every permutation of the vertices of a face maps its points onto its points
        for order in range(2, 9):
            generator = face_quadrature.FaceQuadratureGenerator(order)
            x, y = generator.points.T
            barycentric = np.stack([1 - x - y, x, y], axis=1)
            for permutation in itertools.permutations(range(3)):
                moved = barycentric[:, list(permutation)]
                distance = np.linalg.norm(
                    moved[:, None, :] - barycentric[None, :, :], axis=2
                )
                partner = np.argmin(distance, axis=1)
                self.assertLess(distance.min(axis=1).max(), 1e-14)
                self.assertEqual(sorted(partner), list(range(len(partner))))
                np.testing.assert_allclose(
                    generator.weights[partner], generator.weights, atol=1e-15
                )


if __name__ == "__main__":
    unittest.main()
