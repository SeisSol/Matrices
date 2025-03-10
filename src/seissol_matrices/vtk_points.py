import numpy as np


def vtk_lagrange_2d(d, start=0, divd=None):
    if d < 0:
        return np.empty((0, 2))
    if d == 0:
        return np.array([[1 / 3, 1 / 3]])
    else:
        if divd is None:
            divd = d
        s = 1 / divd
        base = 0
        obase = 0
        allrings = []
        for r in range(start, d // 3 + 1):
            if d == 3 * r:
                allrings += [[s * r, s * r]]
            else:
                ringcorner = [
                    [s * r, s * r],
                    [s * (d - 2 * r), s * r],
                    [s * r, s * (d - 2 * r)],
                ]
                ringborder = [[s * (r + i), s * r] for i in range(1, d - 3 * r)]
                ringborder += [
                    [s * (d - 2 * r - i), s * (r + i)] for i in range(1, d - 3 * r)
                ]
                ringborder += [
                    [s * r, s * (d - 2 * r - i)] for i in range(1, d - 3 * r)
                ]
                allrings += ringcorner + ringborder
        return np.array(allrings)


def vtk_lagrange_3d(d):
    if d == 0:
        return np.array([[1 / 4, 1 / 4, 1 / 4]])
    else:
        s = 1 / d
        base = 0
        obase = 0
        allrings = []
        for r in range(d // 4 + 1):
            if d == 4 * r:
                allrings += [[s * r, s * r, s * r]]
            else:
                ringcorner = [
                    [s * r, s * r, s * r],
                    [s * (d - 3 * r), s * r, s * r],
                    [s * r, s * (d - 3 * r), s * r],
                    [s * r, s * r, s * (d - 3 * r)],
                ]
                ringborder = [[s * (r + i), s * r, s * r] for i in range(1, d - 4 * r)]
                ringborder += [
                    [s * (d - 3 * r - i), s * (r + i), s * r]
                    for i in range(1, d - 4 * r)
                ]
                ringborder += [
                    [s * r, s * (d - 3 * r - i), s * r] for i in range(1, d - 4 * r)
                ]
                ringborder += [[s * r, s * r, s * (r + i)] for i in range(1, d - 4 * r)]
                ringborder += [
                    [s * (d - 3 * r - i), s * r, s * (r + i)]
                    for i in range(1, d - 4 * r)
                ]
                ringborder += [
                    [s * r, s * (d - 3 * r - i), s * (r + i)]
                    for i in range(1, d - 4 * r)
                ]
                subface = vtk_lagrange_2d(d - r, r + 1, d)
                ringfaces = [(i, s * r, j) for i, j in subface]
                ringfaces += [(j, (1 - s * r) - i - j, i) for i, j in subface]
                ringfaces += [(s * r, j, i) for i, j in subface]
                ringfaces += [(j, i, s * r) for i, j in subface]
                allrings += ringcorner + ringborder + ringfaces
        return np.array(allrings)


def vtk_lagrange_2d_from_vtk(d):
    import vtk

    tet = vtk.vtkLagrangeTriangle()
    points = ((d + 1) * (d + 2)) // 2
    tet.GetPointIds().SetNumberOfIds(points)
    tet.GetPoints().SetNumberOfPoints(points)
    tet.Initialize()
    pointlist = []
    for i in range(points):
        barycentric = [0, 0, 0]
        tet.ToBarycentricIndex(i, barycentric)
        pointlist += [[b / d for b in barycentric[:2]]]
    return np.array(pointlist)


def vtk_lagrange_3d_from_vtk(d):
    import vtk

    tet = vtk.vtkLagrangeTetra()
    points = ((d + 1) * (d + 2) * (d + 3)) // 6
    tet.GetPointIds().SetNumberOfIds(points)
    tet.GetPoints().SetNumberOfPoints(points)
    tet.Initialize()
    pointlist = []
    for i in range(points):
        barycentric = [0, 0, 0, 0]
        tet.ToBarycentricIndex(i, barycentric)
        pointlist += [[b / d for b in barycentric[:3]]]
    return np.array(pointlist)
