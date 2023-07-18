import numpy as np
import quad_rules.WitherdenVincentTri
import quad_rules.WitherdenVincentTet
import quad_rules.JaskowiecSukumar
import quad_rules.GaussJacobi
import quad_rules.quadrature

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
            self.geometry = np.array(
                [
                    [0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0],
                ]
            )
            self.face_generator = dg_generator(o, 2)
            n, w = quad_rules.JaskowiecSukumar.JaskowiecSukumar().find_best_rule(
                2 * self.order
            )
        elif self.dim == 2:
            self.generator = basis_functions.BasisFunctionGenerator2D(self.order)
            self.geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            n, w = quad_rules.WitherdenVincentTri.WitherdenVincentTri().find_best_rule(
                2 * self.order
            )
        elif self.dim == 1:
            self.generator = basis_functions.BasisFunctionGenerator1D(self.order)
            self.geometry = np.array([[0.0], [1.0]])
            n, w = quad_rules.GaussJacobi.GaussJacobi(0, 0).find_best_rule(
                2 * self.order
            )
            self.nodes, self.weights = quad_rules.quadrature.transform(
                n, w, self.geometry
            )
        else:
            raise Execption("Can only generate 1D, 2D or 2D basis functions")

        self.nodes, self.weights = quad_rules.quadrature.transform(n, w, self.geometry)
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
                self.M[i, j] = quad_rules.quadrature.quad(
                    self.nodes, self.weights, prod
                )
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
                self.K[dim][i, j] = quad_rules.quadrature.quad(
                    self.nodes, self.weights, prod
                )
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
        # implement Dumbser & Käser, 2006 Table 2 b)
        # chi = x[0,:]
        # tau = x[1,:]
        # y[0,:] = chi ~
        # y[1,:] = tau ~
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
                matrix[i, j] = quad_rules.quadrature.quad(
                    self.face_generator.nodes, self.face_generator.weights, prod
                )
        return matrix

    def fP(self, side):
        return self.face_times_face_mass_matrix(side)

    def volume_to_face_parametrisation(self, x, side):
        # implement Dumbser & Käser, 2006 Table 2 a)
        # chi = x[0,:]
        # tau = x[1,:]
        # y[0,:] = xi(j)
        # y[1,:] = eta(j)
        # y[2,:] = zeta(j)
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
                matrix[i, j] = quad_rules.quadrature.quad(
                    self.face_generator.nodes, self.face_generator.weights, prod
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
    from seissol_matrices import json_io

    # generator = dg_generator(3, 3)
    # xml_io.write_matrix(generator.kDivM(2), "kDivM(2)", "kdivm.xml")
    for order in [2, 7]:
        filename = f"mass_{order}.json"
        face_generator = dg_generator(order, 2)
        mass_2 = face_generator.mass_matrix()
        mass_2_inv = np.linalg.solve(mass_2, np.eye(mass_2.shape[0]))
        json_io.write_matrix(mass_2, "M2", filename)
        json_io.write_matrix(mass_2_inv, "M2inv", filename)

        volume_generator = dg_generator(order, 3)
        mass_3 = volume_generator.mass_matrix()
        num_elements = mass_3.shape[0]
        mass_3_inv = np.linalg.solve(mass_3, np.eye(mass_3.shape[0]))
        rDivM_2 = volume_generator.rDivM(0)
        print(mass_3)
        json_io.write_matrix(mass_3, "M3", filename)
        json_io.write_matrix(mass_3_inv, "M3inv", filename)
        json_io.write_matrix(rDivM_2, "rDivM(0)", f"matrices_{num_elements}.json")
