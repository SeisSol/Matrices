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
        return 1.0, 0.0, 0.0
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


################################################################################


class BasisFunctionGenerator1D(BasisFunctionGenerator):
    def eval_basis(self, x, i):
        r_num = 2 * x - 1.0
        r_den = 1.0
        return singularity_free_jacobi_polynomial(i, 0, 0, r_num, r_den)

    def eval_diff_basis(self, x, i, k):
        if i == 0:
            return 0
        else:
            r_num = 2 * x - 1.0
            r_den = 1.0
            return (
                2
                * (i + 1)
                * singularity_free_jacobi_polynomial(i - 1, 1, 1, r_num, r_den)
            )

    def number_of_basis_functions(self):
        return self.order


################################################################################


class BasisFunctionGenerator2D(BasisFunctionGenerator):
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
