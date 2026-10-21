"""
Writes the matrices SeisSol reads, as JSON files.

    python generate.py <mode> [materialorder] [--basis FAMILY] [--output-dir DIR]

The modes vtk, ader, dr and hom write the matrices of the visualisation, of the
ADER-DG kernels, of dynamic rupture and of the material projections; basis,
faces, degrees and bernstein write what a basis other than the Dubiner one needs
on top (see their functions). --basis selects the basis family (basis_functions.FAMILIES);
the files of a family other than the Dubiner basis go into a directory named after
it, unless --output-dir says otherwise. Every mode writes basis.json there, which
states what the matrices of the family may be relied on for
(basis_functions.describe).
"""

import argparse
import json
import os

import numpy as np

from seissol_matrices import basis_functions, bernstein_matrices, json_io, quad_points
from seissol_matrices import face_quadrature
from seissol_matrices.dg_matrices import FACE_ORIENTATIONS, basis_change, dg_generator
from seissol_matrices.dr_matrices import dr_generator
from seissol_matrices.plasticity_matrices import PlasticityGenerator
from seissol_matrices.vtk_points import vtk_lagrange_2d, vtk_lagrange_3d

ORDERS = range(2, 9)


def vtk(family, output, materialorder):
    vtkpoints2 = {}
    vtkpoints3 = {}
    base = {}
    for deg in range(0, 9):
        vtkpoints2[deg] = vtk_lagrange_2d(deg)
        vtkpoints3[deg] = vtk_lagrange_3d(deg)
        base[f"vtk2d({deg})"] = vtkpoints2[deg].T
        base[f"vtk3d({deg})"] = vtkpoints3[deg].T
    json_io.write_tensors(base, output("vtkbase.json"))
    for basisorder in ORDERS:
        dggen = dg_generator(basisorder, 3, family)
        tensors = {}
        for deg in range(0, 9):
            tensors[f"collvv({basisorder},{deg})"] = dggen.collocate_volume(
                vtkpoints3[deg]
            ).T
            tensors[
                f"collff({basisorder},{deg})"
            ] = dggen.face_generator.collocate_volume(vtkpoints2[deg]).T
            for f in range(4):
                tensors[f"collvf({basisorder},{deg},{f})"] = dggen.collocate_face(
                    vtkpoints2[deg], f
                ).T
        json_io.write_tensors(tensors, output(f"vtko{basisorder}.json"))


def ader(family, output, materialorder):
    for order in ORDERS:
        if materialorder is None:
            filename = output(f"linear-ader-{order}.json")
        else:
            filename = output(f"linear-ader-{order}-h{materialorder}.json")
        dggen = dg_generator(order, 3, family)
        tensors = {}
        for dim in range(3):
            tensors[f"kDivMT({dim})"] = dggen.kDivMT(dim, materialorder)
            tensors[f"kDivM({dim})"] = dggen.kDivM(dim, materialorder)
        for line in range(3):
            tensors[f"fP({line})"] = dggen.fP(line)
        for face in range(4):
            tensors[f"rDivM({face})"] = dggen.rDivM(face, materialorder)
            tensors[f"rT({face})"] = dggen.rT(face)
            tensors[f"fMrT({face})"] = dggen.fMrT(face)
        json_io.write_tensors(tensors, filename)


def dr(family, output, materialorder):
    for order in ORDERS:
        for quadname, quadrule in quad_points.dr_rules(order).items():
            if materialorder is None:
                filename = output(f"dr-{quadname}-{order}.json")
            else:
                filename = output(f"dr-{quadname}-{order}-h{materialorder}.json")
            generator = dr_generator(order, quadrule, family)
            tensors = {
                "quadpoints": generator.quadpoints(),
                "quadweights": generator.quadweights(),
                "resample": generator.resample(),
                "V2QuadTo2m": generator.V2QuadTo2m(),
                "V2mTo2Quad": generator.V2mTo2Quad(),
            }
            for a in range(0, 4):
                for b in range(0, 4):
                    tensors[f"V3mTo2n({a},{b})"] = generator.V3mTo2n(a, b)
                    tensors[f"V3mTo2nTWDivM({a},{b})"] = generator.V3mTo2nTWDivM(
                        a, b, matorder=materialorder
                    )
            if not basis_functions.describe(family)["hierarchical"]:
                # the traces of the time derivatives at the points of the fault
                for degree in range(order - 1):
                    tensors[f"V2mTo2QuadDegree({degree})"] = generator.V2mTo2Quad(
                        degree
                    )
            json_io.write_tensors(tensors, filename)


def hom(family, output, materialorder):
    if materialorder is None:
        raise ValueError("hom needs the material order")
    for mode in ("nb", "ip"):
        generator = PlasticityGenerator(materialorder, family)
        json_io.write_tensors(
            {
                "hompoints": generator.nodes(mode),
                "homproject": generator.generate_Vandermonde_inv(mode),
                "homiproject": generator.generate_Vandermonde(mode),
            },
            output(f"hom-{mode}-h{materialorder}.json"),
        )
        for order in ORDERS:
            for quadname, quadrule in quad_points.dr_rules(order).items():
                generator = dr_generator(materialorder, quadrule, family)
                json_io.write_tensors(
                    {
                        f"homV3mTo2n({a},{b})": generator.V3mTo2n(a, b)
                        for a in range(0, 4)
                        for b in range(0, 4)
                    },
                    output(f"homdr-{mode}-{quadname}-{order}-h{materialorder}.json"),
                )


def basis(family, output, materialorder):
    """
    basis-<order>.json: the integrals of the basis functions (basisIntegrals,
    faceBasisIntegrals), which give the mean of a cell where the first
    coefficient does not, and the changes to and from the Dubiner basis
    (toDubiner: c_dubiner = toDubiner c; fromDubiner; faceToDubiner,
    faceFromDubiner), for checkpoints and for comparing runs.
    """
    for order in ORDERS:
        generator = dg_generator(order, 3, family)
        json_io.write_tensors(
            {
                "basisIntegrals": generator.basis_integrals(),
                "faceBasisIntegrals": generator.face_generator.basis_integrals(),
                "toDubiner": basis_change(order, 3, family, "dubiner"),
                "fromDubiner": basis_change(order, 3, "dubiner", family),
                "faceToDubiner": basis_change(order, 2, family, "dubiner"),
                "faceFromDubiner": basis_change(order, 2, "dubiner", family),
            },
            output(f"basis-{order}.json"),
        )


def faces(family, output, materialorder):
    """
    faces-<order>.json: the orientations a face can have against its canonical
    frame (faceOrientations, the ranks per orientation index), the canonical
    traces and lifts (rTCanonical(face,orientation), rDivMCanonical(face,
    orientation)), and the face basis of every degree up to the one of the scheme
    at the points of the symmetric face rule of the flux
    (V2mToFaceQuadDegree(degree)).
    """
    for order in ORDERS:
        generator = dg_generator(order, 3, family)
        tensors = {"faceOrientations": np.array(FACE_ORIENTATIONS, dtype=float)}
        for face in range(4):
            for index, ranks in enumerate(FACE_ORIENTATIONS):
                tensors[f"rTCanonical({face},{index})"] = generator.canonical_rT(
                    face, ranks
                )
                tensors[f"rDivMCanonical({face},{index})"] = generator.canonical_rDivM(
                    face, ranks
                )
        quadrature = face_quadrature.FaceQuadratureGenerator(order, family)
        for degree in range(order):
            tensors[
                f"V2mToFaceQuadDegree({degree})"
            ] = quadrature.generate_V2mToFaceQuad(degree)
        json_io.write_tensors(tensors, output(f"faces-{order}.json"))


def degrees(family, output, materialorder):
    """
    degrees-<mode>-<order>.json: the basis of every degree up to the one of the
    scheme at the points of a plasticity or material point set (vDegree(degree))
    and the projection back (vInvDegree(degree)), for the time derivatives of
    the predictor, which have lower degrees. The nodal points (nb) are left out:
    they are unisolvent for the degree of the scheme alone.
    """
    for mode in ("ip", "sq", "sq3"):
        for order in ORDERS:
            generator = PlasticityGenerator(order, family)
            tensors = {}
            for degree in range(order):
                tensors[f"vDegree({degree})"] = generator.generate_Vandermonde(
                    mode, degree
                )
                tensors[f"vInvDegree({degree})"] = generator.generate_Vandermonde_inv(
                    mode, degree
                )
            json_io.write_tensors(tensors, output(f"degrees-{mode}-{order}.json"))


def bernstein(family, output, materialorder):
    """
    bernstein-<order>.json: the sparse operators of the Bernstein basis (see
    bernstein_matrices), for the degree m = order - 1 and below:
    derivative(degree,direction), elevation3(degree), elevation2(degree),
    trace(face,orientation,degree) (in the canonical frame, a gather),
    liftFace (L_0) and liftScatter(face,orientation,level), the scatter of the
    level scaled by its coefficient l_j, so that the lift is
    sum_j liftScatter(j) (E^m_(m-j))^T liftFace with the transposed face
    elevations in between.
    """
    if family != "bernstein":
        raise ValueError("the Bernstein operators need --basis bernstein")
    for order in ORDERS:
        m = order - 1
        tensors = {}
        for degree in range(1, m + 1):
            for direction in range(3):
                tensors[
                    f"derivative({degree},{direction})"
                ] = bernstein_matrices.derivative(degree, 3, direction)
            tensors[f"elevation3({degree})"] = bernstein_matrices.elevation(degree, 3)
            tensors[f"elevation2({degree})"] = bernstein_matrices.elevation(degree, 2)
        for face in range(4):
            for index, ranks in enumerate(FACE_ORIENTATIONS):
                for degree in range(m + 1):
                    tensors[
                        f"trace({face},{index},{degree})"
                    ] = bernstein_matrices.trace(degree, face, ranks)
                for level, scale in enumerate(bernstein_matrices.lift_coefficients(m)):
                    rows = bernstein_matrices.lift_scatter_indices(
                        m, face, level, ranks
                    )
                    scatter = np.zeros(
                        (
                            bernstein_matrices.count(m, 3),
                            bernstein_matrices.count(m - level, 2),
                        )
                    )
                    scatter[rows, np.arange(len(rows))] = float(scale)
                    tensors[f"liftScatter({face},{index},{level})"] = scatter
        tensors["liftFace"] = bernstein_matrices.lift_face(m)
        json_io.write_tensors(tensors, output(f"bernstein-{order}.json"))


MODES = {
    "vtk": vtk,
    "ader": ader,
    "dr": dr,
    "hom": hom,
    "basis": basis,
    "faces": faces,
    "degrees": degrees,
    "bernstein": bernstein,
}


def main(arguments=None):
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("mode", choices=sorted(MODES))
    parser.add_argument("materialorder", nargs="?", type=int, default=None)
    parser.add_argument("--basis", default="dubiner", choices=basis_functions.FAMILIES)
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(arguments)
    if args.mode == "hom" and args.materialorder is None:
        parser.error("hom needs the material order")
    if args.mode == "bernstein" and args.basis != "bernstein":
        parser.error("bernstein needs --basis bernstein")

    directory = args.output_dir
    if directory is None:
        directory = "." if args.basis == "dubiner" else args.basis
    os.makedirs(directory, exist_ok=True)
    output = lambda filename: os.path.join(directory, filename)
    with open(output("basis.json"), "w") as f:
        json.dump(basis_functions.describe(args.basis), f, indent=2)
        f.write("\n")
    MODES[args.mode](args.basis, output, args.materialorder)


if __name__ == "__main__":
    main()
