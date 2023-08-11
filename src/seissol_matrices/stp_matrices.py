import numpy as np

from seissol_matrices import basis_functions
from seissol_matrices import dg_matrices

### generates matrices which are needed for the space time predictor:
### Z
### value at 0
### time integrated


class stp_generator:
    def __init__(self, o):
        self.order = o

        self.generator = basis_functions.BasisFunctionGenerator1D(self.order)
        self.dg_generator = dg_matrices.dg_generator(self.order, 1)
        self.S = self.dg_generator.mass_matrix()

    def w(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        w = np.zeros((number_of_basis_functions,))
        for i in range(number_of_basis_functions):
            w[i] = self.generator.eval_basis(0, i)
        return np.reshape(w, (number_of_basis_functions, 1))

    def S(self):
        return self.S

    def K(self):
        return self.dg_generator.stiffness_matrix(0)

    def S_hat(self):
        W = self.W()
        K_t = self.dg_generator.stiffness_matrix(0)
        return np.linalg.solve(W - K_t, self.S)

    def W(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        W = np.zeros((number_of_basis_functions, number_of_basis_functions))

        for i in range(number_of_basis_functions):
            for j in range(number_of_basis_functions):
                W[i, j] = self.generator.eval_basis(1.0, i) * self.generator.eval_basis(
                    1.0, j
                )

        return W

    def Z(self):
        W = self.W()
        K_t = self.dg_generator.stiffness_matrix(0)

        return np.linalg.solve(self.S, W - K_t)

    def time_int(self):
        number_of_basis_functions = self.generator.number_of_basis_functions()
        t = np.zeros((number_of_basis_functions,))
        t[0] = 1

        return np.reshape(t, (number_of_basis_functions, 1))


if __name__ == "__main__":
    from seissol_matrices import json_io

    N = 3
    generator = stp_generator(N)
    filename = f"stp_{N}.json"
    Z = generator.Z()
    print(Z)
    json_io.write_matrix(Z, "Z", filename)

    w = generator.w()
    print(w)
    json_io.write_matrix(np.linalg.solve(generator.S, w), "wHat", filename)
    json_io.write_matrix(generator.time_int(), "timeInt", filename)
