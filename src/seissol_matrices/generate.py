import sys
from seissol_matrices import json_io
from dg_matrices import dg_generator
from vtk_points import vtk_lagrange_2d, vtk_lagrange_3d


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


if __name__ == "__main__":
    main()
