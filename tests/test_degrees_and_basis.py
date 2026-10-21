#!/usr/bin/env python3

import json
import math
import os
import shutil
import tempfile
import unittest

import numpy as np

from seissol_matrices import (
    basis_functions,
    bernstein_matrices,
    dg_matrices,
    generate,
    json_io,
    plasticity_matrices,
)


class test_degrees_and_basis(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(4)

    def test_degree_projection(self):
        # the projection of a degree reproduces its basis; in the hierarchical
        # bases the matrices of a degree are the leading columns and rows
        for family in basis_functions.FAMILIES:
            for mode in ("ip", "sq", "sq3"):
                generator = plasticity_matrices.PlasticityGenerator(4, family)
                v = generator.generate_Vandermonde(mode)
                vInv = generator.generate_Vandermonde_inv(mode)
                for degree in range(4):
                    vd = generator.generate_Vandermonde(mode, degree)
                    vInvd = generator.generate_Vandermonde_inv(mode, degree)
                    count = math.comb(degree + 3, 3)
                    self.assertEqual(vd.shape, (v.shape[0], count))
                    np.testing.assert_allclose(
                        vInvd @ vd, np.eye(count), rtol=0, atol=1e-12
                    )
                    if family != "bernstein":
                        np.testing.assert_array_equal(vd, v[:, :count])
                        np.testing.assert_allclose(
                            vInvd, vInv[:count], rtol=0, atol=1e-13
                        )

    def test_basis_integrals(self):
        for order in (1, 3, 5):
            n3 = math.comb(order + 2, 3)
            n2 = math.comb(order + 1, 2)
            expected = {
                "dubiner": (np.eye(n3)[0] / 6, np.eye(n2)[0] / 2),
                "dubiner-normalized": (
                    np.eye(n3)[0] * math.sqrt(1 / 6),
                    np.eye(n2)[0] * math.sqrt(1 / 2),
                ),
                "bernstein": (np.full(n3, 1 / 6 / n3), np.full(n2, 1 / 2 / n2)),
            }
            for family, (volume, face) in expected.items():
                generator = dg_matrices.dg_generator(order, 3, family)
                np.testing.assert_allclose(
                    generator.basis_integrals(), volume, rtol=0, atol=1e-15
                )
                np.testing.assert_allclose(
                    generator.face_generator.basis_integrals(), face, rtol=0, atol=1e-15
                )

    def test_basis_change(self):
        for dim in (2, 3):
            for order in (1, 4, 6):
                for source in basis_functions.FAMILIES:
                    for target in basis_functions.FAMILIES:
                        T = dg_matrices.basis_change(order, dim, source, target)
                        back = dg_matrices.basis_change(order, dim, target, source)
                        np.testing.assert_allclose(
                            back @ T, np.eye(T.shape[0]), rtol=0, atol=1e-11
                        )
                        points = self.rng.random((dim, 9)) / dim
                        c = self.rng.standard_normal(T.shape[1])
                        np.testing.assert_allclose(
                            basis_functions.generator(target, dim, order).eval_all(
                                points
                            )
                            @ (T @ c),
                            basis_functions.generator(source, dim, order).eval_all(
                                points
                            )
                            @ c,
                            rtol=0,
                            atol=1e-10,
                        )

    def test_describe(self):
        self.assertEqual(basis_functions.describe("dubiner")["mass"], "diagonal")
        self.assertEqual(
            basis_functions.describe("dubiner-normalized")["mass"], "identity"
        )
        bernstein = basis_functions.describe("bernstein")
        self.assertEqual(bernstein["mass"], "dense")
        self.assertFalse(bernstein["hierarchical"])

    def test_generate_arguments(self):
        # a mode that cannot run with its arguments stops before writing anything
        directory = tempfile.mkdtemp()
        try:
            for arguments in (["hom"], ["bernstein", "--basis", "dubiner"]):
                with self.assertRaises(SystemExit):
                    generate.main(arguments + ["--output-dir", directory])
            self.assertEqual(os.listdir(directory), [])
        finally:
            shutil.rmtree(directory)

    def test_generate(self):
        # the modes basis, faces, degrees and bernstein write what they say, for a
        # reduced set of orders
        directory = tempfile.mkdtemp()
        orders = generate.ORDERS
        try:
            generate.ORDERS = range(2, 5)
            for mode in ("basis", "faces", "degrees", "bernstein"):
                generate.main([mode, "--basis", "bernstein", "--output-dir", directory])
            with open(os.path.join(directory, "basis.json")) as f:
                self.assertEqual(json.load(f), basis_functions.describe("bernstein"))
            order = 4
            m = order - 1
            filename = os.path.join(directory, f"bernstein-{order}.json")
            face, index = 2, 3
            ranks = dg_matrices.FACE_ORIENTATIONS[index]
            chain = json_io.read_tensor("liftFace", filename)
            lift = np.zeros((math.comb(m + 3, 3), math.comb(m + 2, 2)))
            for level in range(m + 1):
                if level > 0:
                    chain = (
                        json_io.read_tensor(f"elevation2({m - level + 1})", filename).T
                        @ chain
                    )
                lift += (
                    json_io.read_tensor(
                        f"liftScatter({face},{index},{level})", filename
                    )
                    @ chain
                )
            np.testing.assert_allclose(
                lift, bernstein_matrices.lift(m, face, ranks), rtol=0, atol=1e-12
            )
            np.testing.assert_array_equal(
                json_io.read_tensor(f"trace({face},{index},{m})", filename),
                bernstein_matrices.trace(m, face, ranks),
            )
            faces = os.path.join(directory, f"faces-{order}.json")
            np.testing.assert_array_equal(
                json_io.read_tensor("faceOrientations", faces),
                np.array(dg_matrices.FACE_ORIENTATIONS, dtype=float),
            )
            generator = dg_matrices.dg_generator(order, 3, "bernstein")
            np.testing.assert_allclose(
                json_io.read_tensor(f"rTCanonical({face},{index})", faces),
                generator.canonical_rT(face, ranks),
                rtol=0,
                atol=1e-14,
            )
            for name in (f"basis-{order}.json", f"degrees-sq-{order}.json"):
                self.assertTrue(os.path.exists(os.path.join(directory, name)))
        finally:
            generate.ORDERS = orders
            shutil.rmtree(directory)


if __name__ == "__main__":
    unittest.main()
