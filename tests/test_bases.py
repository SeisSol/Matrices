#!/usr/bin/env python3

import math
import unittest

import numpy as np

from seissol_matrices import basis_functions, dg_matrices, plasticity_matrices


def random_points(rng, dim, count):
    """Points inside the reference element."""
    points = rng.random((dim, count))
    return points / (1.0 + points.sum(axis=0))


def change_to_dubiner(generator, rng):
    """
    The coefficients of the functions of a generator in the Dubiner basis of the
    same degree, from their values at random points; the residual tells whether
    the functions lie in the polynomials of that degree.
    """
    dubiner = basis_functions.generator("dubiner", generator.dim(), generator.order)
    points = random_points(
        rng, generator.dim(), 4 * dubiner.number_of_basis_functions()
    )
    D = dubiner.eval_all(points)
    F = generator.eval_all(points)
    T, *_ = np.linalg.lstsq(D, F, rcond=None)
    return T, np.abs(D @ T - F).max(), dubiner


class test_bases(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(1)

    def test_factory(self):
        for family in basis_functions.FAMILIES:
            for dim in (1, 2, 3):
                generator = basis_functions.generator(family, dim, 4)
                self.assertEqual(generator.family, family)
                self.assertEqual(generator.dim(), dim)
                self.assertEqual(
                    generator.number_of_basis_functions(), math.comb(3 + dim, dim)
                )
        with self.assertRaises(ValueError):
            basis_functions.generator("lagrange", 3, 4)
        with self.assertRaises(ValueError):
            basis_functions.generator("bernstein", 4, 4)

    def test_same_space(self):
        # every family spans the polynomials of degree order - 1, and its
        # derivatives are the ones of those polynomials
        for family in basis_functions.FAMILIES:
            for dim in (1, 2, 3):
                for order in range(1, 7):
                    generator = basis_functions.generator(family, dim, order)
                    T, residual, dubiner = change_to_dubiner(generator, self.rng)
                    self.assertLess(residual, 1e-11, (family, dim, order))
                    self.assertEqual(
                        np.linalg.matrix_rank(T), T.shape[0], (family, dim, order)
                    )
                    points = random_points(self.rng, dim, 9)
                    for k in range(dim):
                        expected = dubiner.eval_diff_all(points, k) @ T
                        np.testing.assert_allclose(
                            generator.eval_diff_all(points, k),
                            expected,
                            rtol=0,
                            atol=1e-9 * max(1.0, np.abs(expected).max()),
                            err_msg=f"{family} {dim} {order} d{k}",
                        )

    def test_eval_all(self):
        for family in basis_functions.FAMILIES:
            for dim in (1, 2, 3):
                generator = basis_functions.generator(family, dim, 4)
                points = random_points(self.rng, dim, 5)
                coordinates = points[0] if dim == 1 else points
                values = generator.eval_all(points)
                for i in range(generator.number_of_basis_functions()):
                    np.testing.assert_array_equal(
                        values[:, i], generator.eval_basis(coordinates, i)
                    )

    def test_mass(self):
        for dim in (1, 2, 3):
            for order in range(1, 7):
                dubiner = dg_matrices.dg_generator(order, dim, "dubiner")
                M = dubiner.mass_matrix()
                norms = [
                    dubiner.generator.norm_squared(i)
                    for i in range(dubiner.generator.number_of_basis_functions())
                ]
                np.testing.assert_allclose(M, np.diag(norms), rtol=0, atol=1e-13)

                normalized = dg_matrices.dg_generator(order, dim, "dubiner-normalized")
                np.testing.assert_allclose(
                    normalized.mass_matrix(), np.eye(len(norms)), rtol=0, atol=1e-13
                )

                bernstein = dg_matrices.dg_generator(order, dim, "bernstein")
                MB = bernstein.mass_matrix()
                np.testing.assert_allclose(MB, MB.T, rtol=0, atol=1e-16)
                self.assertGreater(np.linalg.eigvalsh(MB).min(), 0.0)

    def test_bernstein(self):
        for dim in (1, 2, 3):
            for order in range(1, 7):
                generator = basis_functions.generator("bernstein", dim, order)
                degree = order - 1
                points = random_points(self.rng, dim, 11)
                values = generator.eval_all(points)
                # a partition of unity, nonnegative inside
                np.testing.assert_allclose(values.sum(axis=1), 1.0, rtol=0, atol=1e-14)
                self.assertGreaterEqual(values.min(), 0.0)
                # the function of a vertex is one there and the others vanish
                vertices = np.vstack([np.zeros(dim), np.eye(dim)]).T
                at_vertices = generator.eval_all(vertices)
                for v in range(dim + 1):
                    a = [0] * (dim + 1)
                    a[v] = degree
                    i = generator.index(a)
                    expected = np.zeros(generator.number_of_basis_functions())
                    expected[i] = 1.0
                    np.testing.assert_allclose(at_vertices[v], expected, atol=1e-15)
                # multi-index and position agree
                for i in range(generator.number_of_basis_functions()):
                    self.assertEqual(generator.index(generator.multi_index(i)), i)

    def test_bernstein_order(self):
        # a_dim varies slowest, a_1 fastest; raising a_1 moves to the next index
        self.assertEqual(
            basis_functions.bernstein_indices(2, 2),
            [(0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2)],
        )
        indices = basis_functions.bernstein_indices(3, 3)
        self.assertEqual(indices[:4], [(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        self.assertEqual(indices[-1], (0, 0, 3))
        positions = {a: i for i, a in enumerate(indices)}
        for a in indices:
            if sum(a) < 3:
                self.assertEqual(positions[(a[0] + 1,) + a[1:]], positions[a] + 1)

    def test_operators(self):
        # the DG matrices of every family do what they do for the Dubiner basis:
        # kDivMT is the derivative, rT the trace, rDivM = M^-1 rT^T
        for family in basis_functions.FAMILIES:
            for order in (1, 3, 5):
                generator = dg_matrices.dg_generator(order, 3, family)
                basis = generator.generator
                face_basis = generator.face_generator.generator
                coefficients = self.rng.standard_normal(
                    basis.number_of_basis_functions()
                )
                points = random_points(self.rng, 3, 13)
                for d in range(3):
                    np.testing.assert_allclose(
                        basis.eval_all(points) @ (generator.kDivMT(d) @ coefficients),
                        basis.eval_diff_all(points, d) @ coefficients,
                        rtol=0,
                        atol=1e-10,
                        err_msg=f"{family} {order} kDivMT({d})",
                    )
                face_points = random_points(self.rng, 2, 7)
                M = generator.mass_matrix()
                for f in range(4):
                    volume_points = generator.volume_to_face_parametrisation(
                        face_points, f
                    )
                    np.testing.assert_allclose(
                        face_basis.eval_all(face_points)
                        @ (generator.rT(f) @ coefficients),
                        basis.eval_all(volume_points) @ coefficients,
                        rtol=0,
                        atol=1e-11,
                        err_msg=f"{family} {order} rT({f})",
                    )
                    np.testing.assert_allclose(
                        M @ generator.rDivM(f),
                        generator.rT(f).T,
                        rtol=0,
                        atol=1e-13,
                        err_msg=f"{family} {order} rDivM({f})",
                    )

    def test_plasticity_projection(self):
        # the projection from the points of a quadrature rule reproduces the basis
        for family in basis_functions.FAMILIES:
            for mode in ("ip", "sq", "sq3"):
                generator = plasticity_matrices.PlasticityGenerator(4, family)
                v = generator.generate_Vandermonde(mode)
                vInv = generator.generate_Vandermonde_inv(mode)
                np.testing.assert_allclose(
                    vInv @ v, np.eye(v.shape[1]), rtol=0, atol=1e-12
                )


if __name__ == "__main__":
    unittest.main()
