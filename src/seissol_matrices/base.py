import numpy as np


def material_index_last(tensor):
    """
    Move the material mode index of a tensor from the leading to the
    trailing axis.

    The multilinear forms yield the material basis as the leading axis.
    yateto assigns stride one to the leading index and aligns it to the
    vector width, so the material index belongs at the trailing, slowest
    varying end instead.
    """
    return np.moveaxis(tensor, 0, -1)


def collocate(basis, points):
    # takes a basis object, and a points array
    # of the form: npoints × dim
    # outputs a collocation matrix of the form
    # npoints × nbasis

    # as far as the author of these lines is aware,
    # this module has no broadcasing functionality yet
    # for basis functions. Which is sad.

    assert basis.dim() == points.shape[1]

    nbasis = basis.number_of_basis_functions()

    coll = np.empty((nbasis, points.shape[0]))
    for i in range(nbasis):
        for j in range(points.shape[0]):
            coll[i, j] = basis.eval_basis(points[j, :], i)
    return coll
