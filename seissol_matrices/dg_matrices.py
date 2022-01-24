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
            self.scheme = qp.t3.get_good_scheme(self.order + 1)
            self.geometry = [
                [0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        elif self.dim == 2:
            self.generator = basis_functions.BasisFunctionGenerator2D(self.order)
            self.scheme = qp.t2.get_good_scheme(self.order + 1)
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


if __name__ == "__main__":
    from seissol_matrices import xml_io

    generator = dg_generator(3, 3)
    xml_io.write_matrix(generator.kDivM(2), "kDivM(2)", "kdivm.xml")
