#!/usr/bin/env python3

import os


from seissol_matrices import basis_functions, dg_matrices, json_io, quad_points


class FaceQuadratureGenerator:
    """
    The matrices for a flux evaluated at the points of a quadrature rule on the faces
    of a cell: the fully symmetric rule with positive weights and interior points that
    integrates the product of two face basis functions exactly (degree 2 * (order - 1)).

    The points are given on the reference triangle (0, 0), (1, 0), (0, 1) in the
    coordinates the face basis functions are given in; the weights sum to its area 1/2.
    Since every permutation of the vertices maps the rule onto itself, the two cells
    sharing a face see the same points, only numbered differently.
    """

    def __init__(self, order, family="dubiner"):
        self.bf2_generator = basis_functions.generator(family, 2, order)
        self.dg3_generator = dg_matrices.dg_generator(order, d=3, family=family)

        rule = quad_points.symmetric(2 * (order - 1))
        self.points = rule.points()
        self.weights = rule.weights()
        self.vandermonde = self.generate_V2mToFaceQuad()
        self.rTs = [self.dg3_generator.rT(faceid) for faceid in range(4)]

    def generate_V2mToFaceQuad(self):
        """The face basis functions at the points, shape (points, face basis functions)."""
        return self.bf2_generator.eval_all(self.points.T)

    def generate_VFaceQuadTo2m(self):
        """
        The integrals of the face basis functions against a function given at the
        points, by the rule: the counterpart of V2nTo2m (the mass matrix times the
        interpolation) for a function that is not interpolated but integrated.
        """
        return self.vandermonde.T * self.weights[None, :]

    def generate_V3mToFaceQuad(self, faceid):
        """The volume basis functions at the points of a face, through their traces."""
        return self.vandermonde @ self.rTs[faceid]


def main():
    os.makedirs("output", exist_ok=True)
    for order in range(2, 9):
        generator = FaceQuadratureGenerator(order)
        filename = f"output/face-sq-matrices-{order}.json"
        json_io.write_matrix(generator.points, "faceQuadPoints", filename)
        json_io.write_matrix(generator.weights, "faceQuadWeights", filename)
        json_io.write_matrix(
            generator.generate_V2mToFaceQuad(), "V2mToFaceQuad", filename
        )
        json_io.write_matrix(
            generator.generate_VFaceQuadTo2m(), "VFaceQuadTo2m", filename
        )
        for faceid in range(4):
            json_io.write_matrix(
                generator.generate_V3mToFaceQuad(faceid),
                f"V3mToFaceQuad({faceid})",
                filename,
            )


if __name__ == "__main__":
    main()
