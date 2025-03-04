import sys
from seissol_matrices import json_io
from dg_matrices import dg_generator
from dr_matrices import dr_generator
from vtk_points import vtk_lagrange_2d, vtk_lagrange_3d
from seissol_matrices import quad_points
from plasticity_matrices import PlasticityGenerator


def main():
    assert len(sys.argv) > 1
    if sys.argv[1] == "vtk":
        import dg_matrices

        vtkpoints2 = {}
        vtkpoints3 = {}
        for deg in range(0, 9):
            vtkpoints2[deg] = vtk_lagrange_2d(deg)
            vtkpoints3[deg] = vtk_lagrange_3d(deg)
            json_io.write_matrix(vtkpoints2[deg].T, f"vtk2d({deg})", f"vtkbase.json")
            json_io.write_matrix(vtkpoints3[deg].T, f"vtk3d({deg})", f"vtkbase.json")
        for basisorder in range(2, 9):
            dggen = dg_generator(basisorder, 3)
            for deg in range(0, 9):
                json_io.write_matrix(
                    dggen.collocate_volume(vtkpoints3[deg]).T,
                    f"collvv({basisorder},{deg})",
                    f"vtko{basisorder}.json",
                )
                json_io.write_matrix(
                    dggen.face_generator.collocate_volume(vtkpoints2[deg]).T,
                    f"collff({basisorder},{deg})",
                    f"vtko{basisorder}.json",
                )
                for f in range(4):
                    json_io.write_matrix(
                        dggen.collocate_face(vtkpoints2[deg], f).T,
                        f"collvf({basisorder},{deg},{f})",
                        f"vtko{basisorder}.json",
                    )
    elif sys.argv[1] == "ader":
        materialorder = None
        if len(sys.argv) > 2:
            materialorder = int(sys.argv[2])
        for order in range(2, 9):
            filename = f"linear-ader-{order}-h{materialorder}.json"
            if materialorder is None:
                filename = f"linear-ader-{order}.json"
            dggen = dg_generator(order, 3)
            for dim in range(3):
                json_io.write_tensor(
                    dggen.kDivMT(dim, materialorder),
                    f"kDivMT({dim})",
                    filename,
                )
                json_io.write_tensor(
                    dggen.kDivM(dim, materialorder), f"kDivM({dim})", filename
                )
            for line in range(3):
                json_io.write_tensor(dggen.fP(line), f"fP({line})", filename)
            for face in range(4):
                json_io.write_tensor(
                    dggen.rDivM(face, materialorder), f"rDivM({face})", filename
                )
                json_io.write_tensor(dggen.rT(face), f"rT({face})", filename)
                json_io.write_tensor(dggen.fMrT(face), f"fMrT({face})", filename)
    elif sys.argv[1] == "dr":
        materialorder = None
        if len(sys.argv) > 2:
            materialorder = int(sys.argv[2])
        for order in range(2, 9):
            for quadname, quadrule in zip(
                ["stroud", "dunavant", "witherden_vincent"],
                [
                    quad_points.stroud(order + 1),
                    quad_points.dunavant(order + 1),
                    quad_points.witherden_vincent(order + 1),
                ],
            ):
                filename = f"dr-{quadname}-{order}-h{materialorder}.json"
                if materialorder is None:
                    filename = f"dr-{quadname}-{order}.json"
                generator = dr_generator(order, quadrule)

                quadpoints = generator.quadpoints()
                quadweights = generator.quadweights()
                resample = generator.resample()
                V2QuadTo2m = generator.V2QuadTo2m()
                V2mTo2Quad = generator.V2mTo2Quad()

                json_io.write_matrix(quadpoints, "quadpoints", filename)
                json_io.write_matrix(
                    quadweights.reshape(-1, 1), "quadweights", filename
                )
                json_io.write_matrix(resample, "resample", filename)
                json_io.write_matrix(V2QuadTo2m, "V2QuadTo2m", filename)
                json_io.write_matrix(V2mTo2Quad, "V2mTo2Quad", filename)
                for a in range(0, 4):
                    for b in range(0, 4):
                        V3mTo2n = generator.V3mTo2n(a, b)
                        V3mTo2nTWDivM = generator.V3mTo2nTWDivM(
                            a, b, matorder=materialorder
                        )
                        json_io.write_matrix(V3mTo2n, f"V3mTo2n({a},{b})", filename)
                        json_io.write_matrix(
                            V3mTo2nTWDivM, f"V3mTo2nTWDivM({a},{b})", filename
                        )
    elif sys.argv[1] == "hom":
        materialorder = None
        if len(sys.argv) > 2:
            materialorder = int(sys.argv[2])
        for mode in ("nb", "ip"):
            filename = f"hom-{mode}-h{materialorder}.json"
            generator = PlasticityGenerator(materialorder)
            points = generator.nodes(mode)
            interpolator = generator.generate_Vandermonde_inv(mode)

            json_io.write_matrix(points, "hompoints", filename)
            json_io.write_matrix(interpolator, "homproject", filename)


if __name__ == "__main__":
    main()
