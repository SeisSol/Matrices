import math
from abc import ABC
import numpy as np
import scipy.special as sp_special

# based on the appendix of Josep de la Puente's thesis
# and the singuality free implementation in
# https://github.com/SeisSol/SeisSol/blob/master/src/Numerical_aux/Functions.cpp


def singularity_free_jacobi_polynomial_factors(m, a, b):
    c_0 = 2.0 * m + a + b
    c_1 = c_0 - 1.0
    c_2 = a * a - b * b
    c_3 = c_0 * (c_0 - 2.0)
    c_4 = 2.0 * (m + a - 1.0) * (m + b - 1.0) * c_0
    c_5 = 2.0 * m * (m + a + b) * (c_0 - 2.0)
    return c_1, c_2, c_3, c_4, c_5


def singularity_free_jacobi_polynomial_recursion(x, y, factors, pm_1, pm_2):
    return (
        factors[0] * (factors[1] * y + factors[2] * x) * pm_1
        - factors[3] * y * y * pm_2
    ) / factors[4]


def singularity_free_jacobi_polynomial(n, a, b, x, y):
    if n == 0:
        return np.ones(np.shape(x))

    pm_1 = 1.0
    pm = (0.5 * a - 0.5 * b) * y + (1.0 + 0.5 * (a + b)) * x
    for m in range(2, n + 1):
        pm_2 = pm_1
        pm_1 = pm
        c = singularity_free_jacobi_polynomial_factors(m, a, b)
        pm = singularity_free_jacobi_polynomial_recursion(x, y, c, pm_1, pm_2)
    return pm


def singularity_free_jacobi_polynomial_and_derivatives(n, a, b, x, y):
    if n == 0:
        return np.ones(np.shape(x)), np.zeros(np.shape(x)), np.zeros(np.shape(x))
    pm_1 = 1.0
    ddx_pm_1 = 0.0
    ddy_pm_1 = 0.0
    pm = singularity_free_jacobi_polynomial(1, a, b, x, y)
    ddx_pm = 1.0 + 0.5 * (a + b)
    ddy_pm = 0.5 * (a - b)
    for m in range(2, n + 1):
        pm_2 = pm_1
        pm_1 = pm
        ddx_pm_2 = ddx_pm_1
        ddx_pm_1 = ddx_pm
        ddy_pm_2 = ddy_pm_1
        ddy_pm_1 = ddy_pm
        c = singularity_free_jacobi_polynomial_factors(m, a, b)
        pm = singularity_free_jacobi_polynomial_recursion(x, y, c, pm_1, pm_2)
        ddx_pm = (
            c[0] * (c[2] * pm_1 + (c[1] * y + c[2] * x) * ddx_pm_1)
            - c[3] * y * y * ddx_pm_2
        ) / c[4]
        ddy_pm = (
            c[0] * (c[1] * pm_1 + (c[1] * y + c[2] * x) * ddy_pm_1)
            - c[3] * (2.0 * y * pm_2 + y * y * ddy_pm_2)
        ) / c[4]
    return pm, ddx_pm, ddy_pm


################################################################################


class BasisFunctionGenerator(ABC):
    """
    The basis functions of the polynomials of degree order - 1 on the reference
    element: the unit interval, the triangle (0, 0), (1, 0), (0, 1), or the
    tetrahedron (0, 0, 0), (1, 0, 0), (0, 1, 0), (0, 0, 1).

    A point x is given by its coordinates x[0], ..., x[dim - 1], each a number or
    an array of points; on the interval, by its coordinate itself.

    family: the name a generator of this basis is asked for (see generator)
    hierarchical: whether the functions of degree at most m come first, in the
    same order for every order; then a polynomial of degree m has nonzero
    coefficients in the leading entries alone
    orthogonal: whether the functions are orthogonal in L2 on the reference element
    orthonormal: whether they are, in addition, of unit norm there
    """

    family = None
    hierarchical = False
    orthogonal = False
    orthonormal = False

    def __init__(self, order):
        self.order = order

    # computes phi_x(x)
    def eval_basis(self, x, i):
        pass

    # computes d / d_x_k phi_i (x)
    def eval_diff_basis(self, x, i, k):
        pass

    def number_of_basis_functions(self, o):
        pass

    def dim(self):
        pass

    def degree(self):
        return self.order - 1

    def eval_all(self, x):
        """All basis functions at the points x (dim x points), shape (points, functions)."""
        x = _as_points(x, self.dim())
        n = self.number_of_basis_functions()
        return np.stack(
            [
                np.broadcast_to(self.eval_basis(_coordinates(x), i), x.shape[1:])
                for i in range(n)
            ],
            axis=-1,
        )

    def eval_diff_all(self, x, k):
        """The derivatives d / d_x_k of all basis functions at the points x, shape
        (points, functions)."""
        x = _as_points(x, self.dim())
        n = self.number_of_basis_functions()
        return np.stack(
            [
                np.broadcast_to(
                    self.eval_diff_basis(_coordinates(x), i, k), x.shape[1:]
                )
                for i in range(n)
            ],
            axis=-1,
        )


def _as_points(x, dim):
    """Points as an array of shape (dim, points), from that shape, from the dim
    coordinates of one point, or on the interval from the coordinates of the
    points."""
    x = np.asarray(x, dtype=float)
    if x.ndim <= 1:
        return x.reshape(1, -1) if dim == 1 else x.reshape(dim, 1)
    if x.ndim != 2 or x.shape[0] != dim:
        raise ValueError(f"points of shape {x.shape}, not ({dim}, points)")
    return x


def _coordinates(x):
    """What eval_basis takes for points of shape (dim, points): the coordinate
    itself on the interval, the rows otherwise."""
    return x[0] if x.shape[0] == 1 else x


################################################################################


class BasisFunctionGenerator1D(BasisFunctionGenerator):
    """The shifted Legendre polynomials P_i(2 x - 1) on the unit interval."""

    family = "dubiner"
    hierarchical = True
    orthogonal = True

    def dim(self):
        return 1

    def norm_squared(self, i):
        """The L2 norm of the i-th function squared."""
        return 1.0 / (2 * i + 1)

    def eval_basis(self, x, i):
        r_num = 2 * x - 1.0
        r_den = 1.0
        return singularity_free_jacobi_polynomial(i, 0, 0, r_num, r_den)

    def eval_diff_basis(self, x, i, k):
        if i == 0:
            return np.zeros(x.shape)
        else:
            r_num = 2 * x - 1.0
            r_den = 1.0
            return (i + 1) * singularity_free_jacobi_polynomial(
                i - 1, 1, 1, r_num, r_den
            )

    def number_of_basis_functions(self):
        return self.order


################################################################################


class BasisFunctionGenerator2D(BasisFunctionGenerator):
    """The Dubiner basis on the triangle, ordered by degree."""

    family = "dubiner"
    hierarchical = True
    orthogonal = True

    def dim(self):
        return 2

    def norm_squared(self, i):
        """The L2 norm of the i-th function squared."""
        p, q = self.roll_index(i)
        return 1.0 / (2 * (2 * p + 1) * (p + q + 1))

    def unroll_index(self, i):
        n = i[0] + i[1]
        tri = 0.5 * n * (n + 1)
        return int(tri + i[1])

    def roll_index(self, x):
        n = int(-0.5 + np.sqrt(0.25 + 2 * x))
        tri = int(0.5 * n * (n + 1))
        j = int(x - tri)
        i = int(n - j)
        return (i, j)

    def eval_basis(self, x, i):
        j = self.roll_index(i)
        r_num = 2.0 * x[0] - 1.0 + x[1]
        r_den = 1.0 - x[1]
        s_num = 2.0 * x[1] - 1.0
        s_den = 1.0

        t_i = singularity_free_jacobi_polynomial(j[0], 0, 0, r_num, r_den)
        t_ij = singularity_free_jacobi_polynomial(j[1], 2 * j[0] + 1, 0, s_num, s_den)

        return t_i * t_ij

    def eval_diff_basis(self, x, i, k):
        j = self.roll_index(i)
        r_num = 2.0 * x[0] - 1.0 + x[1]
        r_den = 1.0 - x[1]
        s_num = 2.0 * x[1] - 1.0
        s_den = 1.0

        t_i = singularity_free_jacobi_polynomial_and_derivatives(
            j[0], 0, 0, r_num, r_den
        )
        t_ij = singularity_free_jacobi_polynomial_and_derivatives(
            j[1], 2 * j[0] + 1, 0, s_num, s_den
        )

        d_dalpha = (
            lambda dr_num, dr_den, d_t: (t_i[1] * dr_num + t_i[2] * dr_den) * t_ij[0]
            + t_i[0] * t_ij[1] * d_t
        )

        if k == 0:
            return d_dalpha(2.0, 0.0, 0.0)
        elif k == 1:
            return d_dalpha(1.0, -1.0, 2.0)
        else:
            raise Exception(f"Can't take the derivative in the {k}th direction.")

    def number_of_basis_functions(self):
        return self.order * (self.order + 1) // 2


################################################################################


class BasisFunctionGenerator3D(BasisFunctionGenerator):
    """The Dubiner basis on the tetrahedron, ordered by degree."""

    family = "dubiner"
    hierarchical = True
    orthogonal = True

    def dim(self):
        return 3

    def norm_squared(self, i):
        """The L2 norm of the i-th function squared."""
        p, q, r = self.roll_index(i)
        return 1.0 / (2 * (2 * p + 1) * (p + q + 1) * (2 * p + 2 * q + 2 * r + 3))

    def unroll_index(self, i):
        n = i[0] + i[1] + i[2]
        tet = (n * (n + 1) * (n + 2)) / 6.0
        p = i[2] * (n + 1) - i[2] * (i[2] - 1) / 2
        return int(tet + i[1] + p)

    def roll_index(self, x):
        # find the biggest tetrahedral number smaller than x
        # analytic inversion is horrible
        n = 0
        while (n + 1) * (n + 2) * (n + 3) < 6 * (x + 1):
            n += 1
        tet = (n * (n + 1) * (n + 2)) / 6.0

        k = 0
        while (k + 1) * (n + 1) - k * (k + 1) / 2 < x + 1 - tet:
            k += 1
        p = k * (n + 1) - k * (k - 1) / 2

        j = int(x - tet - p)
        i = int(n - j - k)
        return (i, j, k)

    def eval_basis(self, x, i):
        j = self.roll_index(i)
        r_num = 2.0 * x[0] - 1.0 + x[1] + x[2]
        r_den = 1.0 - x[1] - x[2]
        s_num = 2.0 * x[1] - 1.0 + x[2]
        s_den = 1.0 - x[2]
        t_num = 2.0 * x[2] - 1.0
        t_den = 1.0

        t_i = singularity_free_jacobi_polynomial(j[0], 0, 0, r_num, r_den)
        t_ij = singularity_free_jacobi_polynomial(j[1], 2 * j[0] + 1, 0, s_num, s_den)
        t_ijk = singularity_free_jacobi_polynomial(
            j[2], 2 * j[0] + 2 * j[1] + 2, 0, t_num, 1.0
        )

        return t_i * t_ij * t_ijk

    def eval_diff_basis(self, x, i, k):
        j = self.roll_index(i)
        r_num = 2.0 * x[0] - 1.0 + x[1] + x[2]
        s_num = 2.0 * x[1] - 1.0 + x[2]
        t_num = 2.0 * x[2] - 1.0
        r_den = 1.0 - x[1] - x[2]
        s_den = 1.0 - x[2]
        t_den = 1.0

        t_i = singularity_free_jacobi_polynomial_and_derivatives(
            j[0], 0, 0, r_num, r_den
        )
        t_ij = singularity_free_jacobi_polynomial_and_derivatives(
            j[1], 2 * j[0] + 1, 0, s_num, s_den
        )
        t_ijk = singularity_free_jacobi_polynomial_and_derivatives(
            j[2], 2 * j[0] + 2 * j[1] + 2, 0, t_num, t_den
        )

        d_dalpha = (
            lambda dr_num, dr_den, ds_num, ds_den, dt: (
                t_i[1] * dr_num + t_i[2] * dr_den
            )
            * t_ij[0]
            * t_ijk[0]
            + t_i[0] * (t_ij[1] * ds_num + t_ij[2] * ds_den) * t_ijk[0]
            + t_i[0] * t_ij[0] * (t_ijk[1] * dt)
        )

        if k == 0:
            return d_dalpha(2.0, 0.0, 0.0, 0.0, 0.0)
        elif k == 1:
            return d_dalpha(1.0, -1.0, 2.0, 0.0, 0.0)
        elif k == 2:
            return d_dalpha(1.0, -1.0, 1.0, -1.0, 2.0)
        else:
            raise Exception(f"Can't take the derivative in the {k}th direction.")

    def number_of_basis_functions(self):
        return self.order * (self.order + 1) * (self.order + 2) // 6


################################################################################


class _Normalized:
    """The Dubiner functions divided by their L2 norm on the reference element,
    so that they are orthonormal there."""

    family = "dubiner-normalized"
    orthonormal = True

    def _scale(self, i):
        return 1.0 / np.sqrt(super().norm_squared(i))

    def norm_squared(self, i):
        return 1.0

    def eval_basis(self, x, i):
        return super().eval_basis(x, i) * self._scale(i)

    def eval_diff_basis(self, x, i, k):
        return super().eval_diff_basis(x, i, k) * self._scale(i)


class NormalizedDubinerGenerator1D(_Normalized, BasisFunctionGenerator1D):
    pass


class NormalizedDubinerGenerator2D(_Normalized, BasisFunctionGenerator2D):
    pass


class NormalizedDubinerGenerator3D(_Normalized, BasisFunctionGenerator3D):
    pass


################################################################################


def bernstein_indices(degree, dim):
    """
    The multi-indices (a_1, ..., a_dim) of the Bernstein polynomials of a degree,
    in the order of the basis: a_dim varies slowest and a_1 fastest. The index of
    the vertex at the origin, a_0 = degree - a_1 - ... - a_dim, is implied.

    With this order, raising one a_k by one (or lowering a_0) shifts a run of
    consecutive indices by a fixed offset.
    """

    def at_most(total, count):
        if count == 0:
            return [()]
        return [
            rest + (last,)
            for last in range(total + 1)
            for rest in at_most(total - last, count - 1)
        ]

    return at_most(degree, dim)


def barycentric(x, dim):
    """The barycentric coordinates (l_0, ..., l_dim) of the points x on the
    reference element, l_k = x[k - 1] for k > 0."""
    if dim == 1:
        coordinates = [np.asarray(x, dtype=float)]
    else:
        coordinates = [np.asarray(x[k], dtype=float) for k in range(dim)]
    return [1.0 - sum(coordinates)] + coordinates


class BernsteinGenerator(BasisFunctionGenerator):
    """
    The Bernstein-Bezier basis of degree order - 1 on the reference element,

        B_a(x) = m! / (a_0! ... a_dim!) l_0^a_0 ... l_dim^a_dim,

    with the barycentric coordinates l of x (see barycentric) and the multi-indices
    a of degree m in the order of bernstein_indices.
    """

    family = "bernstein"

    def __init__(self, order, dim):
        super().__init__(order)
        self._dim = dim
        self._indices = bernstein_indices(order - 1, dim)
        self._positions = {a: i for i, a in enumerate(self._indices)}
        self._lower = None

    def dim(self):
        return self._dim

    def number_of_basis_functions(self):
        return len(self._indices)

    def multi_index(self, i):
        """The full multi-index (a_0, ..., a_dim) of the i-th function."""
        a = self._indices[i]
        return (self.degree() - sum(a),) + a

    def index(self, a):
        """The position of the function with the full multi-index a in the basis."""
        assert sum(a) == self.degree()
        return self._positions[tuple(a[1:])]

    def eval_basis(self, x, i):
        a = self.multi_index(i)
        coefficient = math.factorial(self.degree())
        for ak in a:
            coefficient //= math.factorial(ak)
        value = coefficient
        for lk, ak in zip(barycentric(x, self._dim), a):
            value = value * lk**ak
        return value

    def eval_diff_basis(self, x, i, k):
        """d / d_x_k B_a = m (B_(a - e_(k+1)) - B_(a - e_0)), with the functions of
        degree m - 1 and a negative index standing for zero."""
        m = self.degree()
        shape = np.shape(barycentric(x, self._dim)[0])
        if m == 0:
            return np.zeros(shape)
        a = self.multi_index(i)
        if self._lower is None:
            self._lower = BernsteinGenerator(self.order - 1, self._dim)
        lower = self._lower
        value = np.zeros(shape)
        for component, sign in ((k + 1, 1.0), (0, -1.0)):
            if a[component] > 0:
                b = list(a)
                b[component] -= 1
                value = value + sign * m * lower.eval_basis(x, lower.index(b))
        return value


class BernsteinGenerator1D(BernsteinGenerator):
    def __init__(self, order):
        super().__init__(order, 1)


class BernsteinGenerator2D(BernsteinGenerator):
    def __init__(self, order):
        super().__init__(order, 2)


class BernsteinGenerator3D(BernsteinGenerator):
    def __init__(self, order):
        super().__init__(order, 3)


################################################################################

#: The bases a generator can be asked for.
FAMILIES = ("dubiner", "dubiner-normalized", "bernstein")

_GENERATORS = {
    ("dubiner", 1): BasisFunctionGenerator1D,
    ("dubiner", 2): BasisFunctionGenerator2D,
    ("dubiner", 3): BasisFunctionGenerator3D,
    ("dubiner-normalized", 1): NormalizedDubinerGenerator1D,
    ("dubiner-normalized", 2): NormalizedDubinerGenerator2D,
    ("dubiner-normalized", 3): NormalizedDubinerGenerator3D,
    ("bernstein", 1): BernsteinGenerator1D,
    ("bernstein", 2): BernsteinGenerator2D,
    ("bernstein", 3): BernsteinGenerator3D,
}


def generator(family, dim, order):
    """The generator of a basis family on the reference element of a dimension,
    for polynomials of degree order - 1."""
    if family not in FAMILIES:
        raise ValueError(f"unknown basis {family}; known are {', '.join(FAMILIES)}")
    if (family, dim) not in _GENERATORS:
        raise ValueError(f"no basis on a reference element of dimension {dim}")
    return _GENERATORS[(family, dim)](order)
