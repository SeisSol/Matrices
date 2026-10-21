#!/usr/bin/env python3

import unittest

import numpy as np

from seissol_matrices import dg_matrices


class test_zeros(unittest.TestCase):
    def test_snap_zeros(self):
        tensor = np.array([[2.0, 1e-13], [-1.5e-12, -2.5e-12]])
        np.testing.assert_array_equal(
            dg_matrices.snap_zeros(tensor), [[2.0, 0.0], [0.0, -2.5e-12]]
        )
        np.testing.assert_array_equal(
            dg_matrices.snap_zeros(np.zeros((2, 3))), np.zeros((2, 3))
        )

    def test_orthogonal_basis(self):
        # the mass matrices of the orthogonal basis are diagonal, and what is
        # computed from them has its zeros where the integrals have theirs
        for order in range(2, 6):
            generator = dg_matrices.dg_generator(order, 3)
            for M in (generator.mass_matrix(), generator.face_generator.mass_matrix()):
                self.assertTrue(np.all(M == np.diag(np.diag(M))))
            for d in range(3):
                np.testing.assert_array_equal(
                    generator.kDivM(d) != 0, generator.stiffness_matrix(d) != 0
                )
            for side in range(4):
                np.testing.assert_array_equal(
                    generator.rDivM(side) != 0, generator.fMrT(side).T != 0
                )


if __name__ == "__main__":
    unittest.main()
