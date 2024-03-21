import numpy as np
import quad_rules.WitherdenVincentTri
import quad_rules.WitherdenVincentTet
import quad_rules.JaskowiecSukumar
import quad_rules.GaussJacobi
import quad_rules.Quadrature

import base

from seissol_matrices import basis_functions


### Generates matrices which are of general interest for DG methods:
### Mass and Stiffness matrices
### kDivM and kDivMT which are SeisSol specific
class dg_generator:
    def __init__(self, o, d):
        self.order = o
        self.dim = d

        self.M = None
        self.K = self.dim * [None]

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
            self.quadrule_finder = quad_rules.JaskowiecSukumar.JaskowiecSukumar().find_best_rule
            self.generator_finder = basis_functions.BasisFunctionGenerator3D
        elif self.dim == 2:
            self.generator = basis_functions.BasisFunctionGenerator2D(self.order)
            self.geometry = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            self.quadrule_finder = quad_rules.WitherdenVincentTri.WitherdenVincentTri().find_best_rule
            self.generator_finder = basis_functions.BasisFunctionGenerator2D
        elif self.dim == 1:
            self.generator = basis_functions.BasisFunctionGenerator1D(self.order)
            self.geometry = np.array([[0.0], [1.0]])
            self.quadrule_finder = quad_rules.GaussJacobi.GaussJacobi(0, 0).find_best_rule
            self.generator_finder = basis_functions.BasisFunctionGenerator1D
        else:
            raise Execption("Can only generate 1D, 2D or 2D basis functions")

        self.nodes, self.weights = self.get_quadrature_rule(2 * self.order)

    def get_quadrature_rule(self, order):
        n, w = self.quadrule_finder(order)        

        return quad_rules.Quadrature.transform(n, w, self.geometry)

    def multilinear_form(self, face, der, order = None, side = None):
        """
        Given an array der, we compute a tensor V[i...] of dimension
        len(der) defined by (given in C++ parameter pack notation)

        V[i...] = integral(1 * ... * f_i^(der))

        For example, to obtain a mass matrix, choose der = [None,None].
        For a stiffness matrix in x-direction, choose der = [0,None].
        For a trilinear form, choose der = [None,None,None].

        Optionally can supply different orders for the basis functions. (using the order parameter)
        If not given, the order will assumed to be self.order for all constituents.

        Moreover, we can opt to take a projected face basis instead of a volume basis via the
        side argument. If an entry is a negative number there, we assume a non-rotated face,
        otherwise the respective face side. If an entry is None, then we assume the respective volume term.
        """

        # NOTE: we could extract the multiplication/quadrature part from the function selection part.
        # I.e., eval_functions (or a list of basis functions here) could be given as an input parameter,
        # instead of the two current ones.

        # we only support a first derivative at the moment
        assert np.all([d in (None,*(i for i in range(self.dim))) for d in der])

        if order is None:
            order = [self.order] * len(der)
        
        if side is None:
            side = [-1] * len(der)
        
        assert len(order) == len(der)
        assert len(side) == len(der)

        def get_basis_function(o, d, s):
            if s is None:
                basis = self.generator_finder(o)
            else:
                basis = self.face_generator.generator_finder(o)
            basiseval_pre = basis.eval_basis if d is None else lambda x,i: basis.eval_diff_basis(x,i,d)
            if face:
                if s is None:
                    basiseval = lambda x, i: basiseval_pre(self.volume_to_face_parametrisation(x, face), i)
                elif s < 0:
                    basiseval = basiseval_pre
                else:
                    basiseval = lambda x, i: basiseval_pre(self.face_to_face_parametrisation(x, face), i)
            else:
                # volume * face does not make sense when evaluating on the whole volume
                assert s is None
                basiseval = basiseval_pre
            return basiseval, basis.number_of_basis_functions()

        basis_functions = [get_basis_function(o,d,s) for o,d,s in zip(order, der, side)]
        
        if face:
            nodes, weights = self.face_generator.get_quadrature_rule(np.prod(order))
        else:
            nodes, weights = self.get_quadrature_rule(np.prod(order))
        sizes = [bn for _,bn in basis_functions]
        tensor = np.empty(sizes)
        generic_eval_basis = lambda x, index: np.prod(
            basis_functions[k][0](x, i) for k,i in enumerate(index)
        )
        for index in itertools.product(range(size) for size in sizes):
            eval_basis = lambda x: generic_eval_basis(x, index)
            tensor[*index] = quad_rules.Quadrature.quad(nodes, weights, eval_basis)
        return tensor

    def mass_matrix(self):
        if not np.any(self.M == None):
            return self.M
        
        self.M = self.multilinear_form(False, [None, None])
        return self.M

    def stiffness_matrix(self, dim):
        if not np.any(self.K[dim] == None):
            return self.K[dim]
        
        self.K[dim] = self.multilinear_form(False, [dim, None])
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
        return self.multilinear_form(True, [None, None], side=[-1, side])

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
        return self.multilinear_form(True, [None, None], side=[None, side])

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
    
    def collocate_volume(self, points):
        return base.collocate(self.generator, points)
    
    def collocate_face(self, points, side):
        # points are meant to be 2D here

        # this method wants dim × npoints; but points is given the other way. So, we transpose it twice
        projected = self.volume_to_face_parametrisation(points.T, side).T
        return base.collocate(self.generator, projected)

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
