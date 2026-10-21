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
    """
    The basis functions at the points (points x dim), shape functions x points.
    """
    assert basis.dim() == points.shape[1]
    return basis.eval_all(points.T).T
