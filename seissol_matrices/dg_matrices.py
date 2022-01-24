import numpy as np
import quadpy as qp

from seissol_matrices import basis_functions

### Generates matrices which are of general interest for DG methods:
### Mass and Stiffness matrices
### kDivM and kDivMT which are SeisSol specific
class dg_generator:
    def __init__(self, o, d):
        self.order = o
        self.dim = d

        if self.dim == 3:
            self.generator = basis_functions.BasisFunctionGenerator3D(self.order)
            self.scheme = qp.t3.get_good_scheme(self.order * 2)
            self.geometry = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
            self.face_generator = dg_generator(o, 2)
        elif self.dim == 2:
            self.generator = basis_functions.BasisFunctionGenerator2D(self.order)
            self.scheme = qp.t2.get_good_scheme(self.order * 2)
            self.geometry = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
        elif self.dim == 1:
            self.generator = basis_functions.BasisFunctionGenerator1D(self.order)
            self.scheme = qp.c1.gauss_legendre(self.order + 1)
            self.geometry = [0.0, 1.0]
        else:
            raise Execption("Can only generate 1D, 2D or 2D basis functions")

        self.M = None
        self.K = self.dim * [None]

    def mass_matrix(self):
        if not np.any(self.M == None):
            return self.M
        number_of_basis_functions = self.generator.number_of_basis_functions()
        self.M = np.zeros((number_of_basis_functions, number_of_basis_functions))

        for i in range(number_of_basis_functions):
            for j in range(number_of_basis_functions):
                prod = lambda x: self.generator.eval_basis(
                    x, i
                ) * self.generator.eval_basis(x, j)
                self.M[i, j] = self.scheme.integrate(prod, self.geometry)
        return self.M

    def stiffness_matrix(self, dim):
        if not np.any(self.K[dim] == None):
            return self.K[dim]
        number_of_basis_functions = self.generator.number_of_basis_functions()
        self.K[dim] = np.zeros((number_of_basis_functions, number_of_basis_functions))

        for i in range(number_of_basis_functions):
            for j in range(number_of_basis_functions):
                prod = lambda x: self.generator.eval_diff_basis(
                    x, i, dim
                ) * self.generator.eval_basis(x, j)
                self.K[dim][i, j] = self.scheme.integrate(prod, self.geometry)
        return self.K[dim]

    def kDivM(self, dim):
        stiffness = self.stiffness_matrix(dim)
        mass = self.mass_matrix()

        return np.linalg.solve(mass, stiffness)

    def kDivMT(self, dim):
        stiffness = self.stiffness_matrix(dim)
        mass = self.mass_matrix()

        return np.linalg.solve(mass, stiffness.T)

    def face_to_face_parametrisation(self, x, side):
        y = np.zeros((x.shape[0], x.shape[1]))
        if side == 0:
            y[0, :] = x[1, :]
            y[1, :] = x[0, :]
        elif side == 1:
            y[0, :] = 1 - x[0, :] - x[1, :]
            y[1, :] = x[1, :]
        elif side == 2:
            y[0, :] = x[0, :]
            y[1, :] = 1 - x[0, :] - x[1, :]
        return y

    def face_times_face_mass_matrix(self, side):
        assert self.dim == 3
        number_of_face_basis_functions = (
            self.face_generator.generator.number_of_basis_functions()
        )

        projected_basis_function = (
            lambda x, i: self.face_generator.generator.eval_basis(
                self.face_to_face_parametrisation(x, side), i
            )
        )

        matrix = np.zeros(
            (number_of_face_basis_functions, number_of_face_basis_functions)
        )
        for i in range(number_of_face_basis_functions):
            for j in range(number_of_face_basis_functions):
                prod = lambda x: self.face_generator.generator.eval_basis(
                    x, i
                ) * projected_basis_function(x, j)
                matrix[i, j] = self.face_generator.scheme.integrate(
                    prod, self.face_generator.geometry
                )
        return matrix

    def fP(self, side):
        return self.face_times_face_mass_matrix(side)

    def volume_to_face_parametrisation(self, x, side):
        y = np.zeros((x.shape[0] + 1, x.shape[1]))
        if side == 0:
            y[0, :] = x[1, :]
            y[1, :] = x[0, :]
        elif side == 1:
            y[0, :] = x[0, :]
            y[2, :] = x[1, :]
        elif side == 2:
            y[1, :] = x[1, :]
            y[2, :] = x[0, :]
        elif side == 3:
            y[0, :] = 1 - x[0, :] - x[1, :]
            y[1, :] = x[0, :]
            y[2, :] = x[1, :]
        return y

    def volume_times_face_mass_matrix(self, side):
        assert self.dim == 3
        number_of_face_basis_functions = (
            self.face_generator.generator.number_of_basis_functions()
        )

        projected_basis_function = lambda x, i: self.generator.eval_basis(
            self.volume_to_face_parametrisation(x, side), i
        )
        number_of_basis_functions = self.generator.number_of_basis_functions()

        matrix = np.zeros((number_of_face_basis_functions, number_of_basis_functions))
        for i in range(number_of_face_basis_functions):
            for j in range(number_of_basis_functions):
                prod = lambda x: self.face_generator.generator.eval_basis(
                    x, i
                ) * projected_basis_function(x, j)
                matrix[i, j] = self.face_generator.scheme.integrate(
                    prod, self.face_generator.geometry
                )
        return matrix

    def fMrT(self, side):
        return self.volume_times_face_mass_matrix(side)

    def rT(self, side):
        matrix = self.volume_times_face_mass_matrix(side)
        mass = self.face_generator.mass_matrix()
        return np.linalg.solve(mass, matrix)

    def rDivM(self, side):
        assert self.dim == 3
        matrix = self.rT(side)
        mass = self.mass_matrix()
        return np.linalg.solve(mass, matrix.T)


if __name__ == "__main__":
    from seissol_matrices import xml_io

    generator = dg_generator(3, 3)
    xml_io.write_matrix(generator.kDivM(2), "kDivM(2)", "kdivm.xml")
