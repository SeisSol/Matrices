#!/usr/bin/env python3

import unittest
from fractions import Fraction

import numpy as np

from seissol_matrices import basis_functions, bernstein_matrices, dg_matrices


def random_points(rng, dim, count):
    points = rng.random((dim, count))
    return points / (1.0 + points.sum(axis=0))


def bernstein(dim, degree):
    return basis_functions.generator("bernstein", dim, degree + 1)


class test_bernstein_matrices(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(2)

    def test_face_vertices(self):
        self.assertEqual(
            [dg_matrices.face_vertices(f) for f in range(4)],
            [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)],
        )
        self.assertEqual(
            [dg_matrices.opposite_vertex(f) for f in range(4)], [3, 2, 1, 0]
        )

    def test_derivative(self):
        for dim in (1, 2, 3):
            for degree in range(1, 8):
                upper, lower = bernstein(dim, degree), bernstein(dim, degree - 1)
                points = random_points(self.rng, dim, 9)
                c = self.rng.standard_normal(upper.number_of_basis_functions())
                for k in range(dim):
                    entries = bernstein_matrices.derivative_entries(degree, dim, k)
                    # two entries per row, +m and -m
                    self.assertEqual(
                        sorted(set(entries.values())), [-degree, degree], (dim, k)
                    )
                    D = bernstein_matrices.derivative(degree, dim, k)
                    self.assertEqual(
                        len(entries), 2 * lower.number_of_basis_functions()
                    )
                    np.testing.assert_allclose(
                        lower.eval_all(points) @ (D @ c),
                        upper.eval_diff_all(points, k) @ c,
                        rtol=0,
                        atol=1e-10,
                    )

    def test_elevation(self):
        for dim in (1, 2, 3):
            for degree in range(1, 8):
                upper, lower = bernstein(dim, degree), bernstein(dim, degree - 1)
                E = bernstein_matrices.elevation(degree, dim)
                np.testing.assert_allclose(E.sum(axis=1), 1.0, rtol=0, atol=1e-15)
                points = random_points(self.rng, dim, 9)
                c = self.rng.standard_normal(lower.number_of_basis_functions())
                np.testing.assert_allclose(
                    upper.eval_all(points) @ (E @ c),
                    lower.eval_all(points) @ c,
                    rtol=0,
                    atol=1e-12,
                )

    def test_trace(self):
        for degree in range(0, 8):
            generator = dg_matrices.dg_generator(degree + 1, 3, "bernstein")
            for f in range(4):
                G = bernstein_matrices.trace(degree, f)
                self.assertTrue(np.all(G.sum(axis=1) == 1))
                np.testing.assert_allclose(G, generator.rT(f), rtol=0, atol=1e-12)

    def test_face_permutation(self):
        for degree in range(0, 8):
            generator = dg_matrices.dg_generator(degree + 1, 3, "bernstein")
            Mf = generator.face_generator.mass_matrix()
            for h in range(3):
                P = bernstein_matrices.face_permutation(degree, h)
                self.assertTrue(np.all(P.sum(axis=0) == 1))
                self.assertTrue(np.all(P.sum(axis=1) == 1))
                np.testing.assert_allclose(
                    P, np.linalg.solve(Mf, generator.fP(h)), rtol=0, atol=1e-11
                )

    def test_lift(self):
        self.assertEqual(
            bernstein_matrices.lift_coefficients(3),
            [Fraction(1), Fraction(-3, 2), Fraction(1), Fraction(-1, 4)],
        )
        for degree in range(0, 8):
            generator = dg_matrices.dg_generator(degree + 1, 3, "bernstein")
            M = generator.mass_matrix()
            Mf = generator.face_generator.mass_matrix()
            for f in range(4):
                dense = np.linalg.solve(M, generator.rT(f).T @ Mf)
                np.testing.assert_allclose(
                    bernstein_matrices.lift(degree, f),
                    dense,
                    rtol=0,
                    atol=2e-11 * np.abs(dense).max(),
                    err_msg=f"degree {degree} face {f}",
                )


if __name__ == "__main__":
    unittest.main()
