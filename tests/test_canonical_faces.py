#!/usr/bin/env python3

import itertools
import unittest

import numpy as np

from seissol_matrices import (
    basis_functions,
    bernstein_matrices,
    dg_matrices,
    dr_matrices,
    face_quadrature,
    quad_points,
)


def project(generator, vertices, polynomial):
    """The coefficients of a polynomial given in physical space on the cell with
    the vertices (4 x 3), in the basis of the generator."""
    nodes, weights = generator.get_quadrature_rule(2 * generator.order)
    reference = nodes.T
    jacobian = (vertices[1:] - vertices[0]).T
    physical = vertices[0][:, None] + jacobian @ reference
    values = generator.generator.eval_all(reference)
    moments = (values * weights[:, None]).T @ polynomial(physical)
    return np.linalg.solve(generator.mass_matrix(), moments)


def face_of(cell, shared):
    """The local face of a cell (local vertex -> global vertex) whose vertices are
    the shared global vertices, and its orientation against the canonical frame."""
    for face in range(4):
        globals_ = [cell[v] for v in dg_matrices.face_vertices(face)]
        if set(globals_) == set(shared):
            ordered = sorted(globals_)
            return face, tuple(ordered.index(g) for g in globals_)
    raise AssertionError("no shared face")


class test_canonical_faces(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(3)

    def test_sorted_numbering(self):
        # a cell with its vertices ordered by global ID sees faces 1 and 3 in the
        # canonical frame and faces 0 and 2 with their two last vertices swapped
        self.assertEqual(
            [dg_matrices.sorted_vertex_ranks(f) for f in range(4)],
            [(0, 2, 1), (0, 1, 2), (0, 2, 1), (0, 1, 2)],
        )
        for family in basis_functions.FAMILIES:
            generator = dg_matrices.dg_generator(4, 3, family)
            swap = np.linalg.solve(
                generator.face_generator.mass_matrix(), generator.fP(0)
            )
            for f in range(4):
                expected = generator.rT(f) if f in (1, 3) else swap @ generator.rT(f)
                np.testing.assert_allclose(
                    generator.canonical_rT(f, dg_matrices.sorted_vertex_ranks(f)),
                    expected,
                    rtol=0,
                    atol=1e-12,
                )

    def test_two_cells(self):
        # two cells sharing a face, numbered at random, give the same canonical
        # trace of a polynomial defined on both
        for family in basis_functions.FAMILIES:
            for order in (1, 3, 5):
                generator = dg_matrices.dg_generator(order, 3, family)
                degree = order - 1
                for _ in range(4):
                    shared = self.rng.random((3, 3))
                    normal = np.cross(shared[1] - shared[0], shared[2] - shared[0])
                    apexes = [
                        shared.mean(axis=0) + normal,
                        shared.mean(axis=0) - normal,
                    ]
                    ids = self.rng.permutation(5)
                    exponents = [
                        e for e in itertools.product(range(degree + 1), repeat=3)
                    ]
                    exponents = [e for e in exponents if sum(e) <= degree]
                    weights = self.rng.standard_normal(len(exponents))
                    polynomial = lambda x: sum(
                        w * x[0] ** e[0] * x[1] ** e[1] * x[2] ** e[2]
                        for w, e in zip(weights, exponents)
                    )
                    traces = []
                    for side, apex in enumerate(apexes):
                        points = np.vstack([shared, apex])
                        global_ids = [ids[0], ids[1], ids[2], ids[3 + side]]
                        order_of_cell = self.rng.permutation(4)
                        cell = [global_ids[v] for v in order_of_cell]
                        vertices = points[order_of_cell]
                        face, ranks = face_of(cell, ids[:3])
                        coefficients = project(generator, vertices, polynomial)
                        traces.append(
                            generator.canonical_rT(face, ranks) @ coefficients
                        )
                    np.testing.assert_allclose(
                        traces[0],
                        traces[1],
                        rtol=0,
                        atol=1e-9 * max(1.0, np.abs(traces[0]).max()),
                        err_msg=f"{family} order {order}",
                    )

    def test_bernstein_canonical(self):
        for degree in range(0, 7):
            generator = dg_matrices.dg_generator(degree + 1, 3, "bernstein")
            M = generator.mass_matrix()
            Mf = generator.face_generator.mass_matrix()
            for f in range(4):
                for ranks in dg_matrices.FACE_ORIENTATIONS:
                    canonical = generator.canonical_rT(f, ranks)
                    np.testing.assert_allclose(
                        bernstein_matrices.trace(degree, f, ranks),
                        canonical,
                        rtol=0,
                        atol=1e-12,
                    )
                    dense = np.linalg.solve(M, canonical.T @ Mf)
                    np.testing.assert_allclose(
                        bernstein_matrices.lift(degree, f, ranks),
                        dense,
                        rtol=0,
                        atol=2e-11 * np.abs(dense).max(),
                    )

    def test_dr_face_route(self):
        # the fault reads the volume at its points through the trace: V3mTo2n(a, b)
        # is V2mTo2Quad P_b rT(a), with P_b the reparametrisation b - 1 of the face
        # (none for b = 0), and V3mTo2nTWDivM(a, b) is rDivM(a) P_b^T M_face V2QuadTo2m
        for family in basis_functions.FAMILIES:
            for order in (3, 5):
                rule = quad_points.stroud(order + 1)
                dr = dr_matrices.dr_generator(order, rule, family)
                generator = dr.dg3_generator
                Mf = generator.face_generator.mass_matrix()
                reparametrisations = [np.eye(Mf.shape[0])] + [
                    np.linalg.solve(Mf, generator.fP(h)) for h in range(3)
                ]
                V = dr.V2mTo2Quad()
                project_back = Mf @ dr.V2QuadTo2m()
                for a in range(4):
                    for b in range(4):
                        P = reparametrisations[b]
                        np.testing.assert_allclose(
                            dr.V3mTo2n(a, b),
                            V @ P @ generator.rT(a),
                            rtol=0,
                            atol=1e-12,
                        )
                        np.testing.assert_allclose(
                            dr.V3mTo2nTWDivM(a, b),
                            generator.rDivM(a) @ P.T @ project_back,
                            rtol=0,
                            atol=1e-10,
                        )

    def test_dr_canonical_route(self):
        # with the points of the fault in the canonical frame of the face,
        # V2mTo2Quad reads the canonical trace of either side as the volume basis
        # at the points mapped into the cell, and the flux goes back through
        # V2QuadTo2m and the canonical lift as through V3mTo2nTWDivM
        for family in basis_functions.FAMILIES:
            for order in (3, 5):
                rule = quad_points.stroud(order + 1)
                dr = dr_matrices.dr_generator(order, rule, family)
                generator = dr.dg3_generator
                Mf = generator.face_generator.mass_matrix()
                V = dr.V2mTo2Quad()
                back = Mf @ dr.V2QuadTo2m()
                for f in range(4):
                    for ranks in dg_matrices.FACE_ORIENTATIONS:
                        local = dg_matrices.canonical_to_face_parametrisation(
                            rule.points().T, ranks
                        )
                        volume = generator.generator.eval_all(
                            dg_matrices.volume_to_face_parametrisation(local, f)
                        )
                        np.testing.assert_allclose(
                            V @ generator.canonical_rT(f, ranks),
                            volume,
                            rtol=0,
                            atol=1e-11,
                        )
                        np.testing.assert_allclose(
                            generator.canonical_rDivM(f, ranks) @ back,
                            np.linalg.solve(
                                generator.mass_matrix(),
                                volume.T * rule.weights()[None, :],
                            ),
                            rtol=0,
                            atol=1e-9,
                        )

    def test_degrees(self):
        # the face basis of a lower degree at the points: for a hierarchical basis
        # the leading columns, otherwise a basis of its own
        rule = quad_points.stroud(6)
        for family in basis_functions.FAMILIES:
            dr = dr_matrices.dr_generator(5, rule, family)
            face = face_quadrature.FaceQuadratureGenerator(5, family)
            for degree in range(5):
                for full, lower in (
                    (dr.V2mTo2Quad(), dr.V2mTo2Quad(degree)),
                    (
                        face.generate_V2mToFaceQuad(),
                        face.generate_V2mToFaceQuad(degree),
                    ),
                ):
                    count = (degree + 1) * (degree + 2) // 2
                    self.assertEqual(lower.shape, (full.shape[0], count))
                    if family != "bernstein":
                        np.testing.assert_array_equal(lower, full[:, :count])


if __name__ == "__main__":
    unittest.main()
