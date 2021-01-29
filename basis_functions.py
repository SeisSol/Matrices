from abc import ABC
import numpy as np
import scipy.special as sp_special
#based on the appendix of Josep de la Puente's thesis

def theta_a(x, i):
    return sp_special.jacobi(i, 0, 0)(x)

def diff_theta_a(x, i):
    if i == 0:
        return 0
    else:
        return 0.5*(i+1) * sp_special.jacobi(i-1, 1, 1)(x)
    return 1

def theta_b(x, i, j):
    return ((1.-x)/2.)**i * sp_special.jacobi(j, 2*i+1, 0)(x)

def diff_theta_b(x, i, j):
    if i == 0 and j == 0:
        return 0
    elif i == 0:
        return 0.5*(j + 2) * sp_special.jacobi(j-1, 2, 1)(x)
    elif j == 0:
        return -0.5*i * ((1.-x)/2.)**(i-1)
    else:
        return -0.5*i * ((1.-x)/2.)**(i-1) * sp_special.jacobi(j, 2*i+1, 0)(x) + \
                ((1.-x)/2.)**i * 0.5*(j + 2*i + 2) * sp_special.jacobi(j-1, 2*i+2, 1)(x)

def theta_c(x, i, j, k):
    return ((1.-x)/2.)**(i+j) * sp_special.jacobi(k, 2*i+2*j+2, 0)(x)

def diff_theta_c(x, i, j, k):
    if i+j == 0 and k == 0:
        return 0
    elif i+j == 0:
        return 0.5*(k + 3) * sp_special.jacobi(k-1, 3, 1)(x)
    elif k == 0:
        return -0.5*(i+j) * ((1.-x)/2.)**(i+j-1)
    else:
        return -0.5*(i+j) * ((1.-x)/2.)**(i+j-1) * sp_special.jacobi(k, 2*i+2*j+2, 0)(x) + \
                ((1.-x)/2.)**(i+j) * 0.5*(k+ 2*j + 2*i + 3) * sp_special.jacobi(k-1, 2*i+2*j+3, 1)(x)


################################################################################

class BasisFunctionGenerator(ABC):
    def __init__(self, order):
        self.order = order

    #computes phi_x(x)
    def eval_basis(x, i):
        pass

    #computes d / d_x_k phi_i (x)
    def eval_diff_basis(x, i, k):
        pass

    def number_of_basis_functions(o):
        pass

################################################################################

class BasisFunctionGenerator1D(BasisFunctionGenerator):

    def coordinate_transformation(self, x):
        return 2*x - 1

    def eval_basis(self, x, i):
        r = self.coordinate_transformation(x)
        return theta_a(r, i)

    def eval_diff_basis(self, x, i, k):
        #it's a bit easier here :D
        r = self.coordinate_transformation(x)
        return 2*diff_theta_a(r, i)

    def number_of_basis_functions(self):
        return self.order

################################################################################

class BasisFunctionGenerator2D(BasisFunctionGenerator):

    def coordinate_transformation(self, x):
        r = np.divide(2*x[0], 1-x[1]) - 1
        s = 2*x[1]-1
        return (r, s)

    def unroll_index(self, i):
        n = i[0]+i[1]
        tri = 0.5 * n * (n+1)
        return int(tri + i[1])
    
    def roll_index(self, x):
        n = int(-0.5+np.sqrt(0.25 + 2*x))
        tri = int(0.5 * n * (n+1))
        j = int(x - tri)
        i = int(n - j)
        return(i, j)

    def eval_basis(self, x, i):
        j = self.roll_index(i)
        r, s = self.coordinate_transformation(x)
        return theta_a(r, j[0]) * theta_b(s, j[0], j[1])

    def number_of_basis_functions(self):
        return self.order * (self.order+1) // 2

################################################################################

class BasisFunctionGenerator3D(BasisFunctionGenerator):

    def coordinate_transformation(self, x):
        r = np.divide(2*x[0], 1-x[1]-x[2]) - 1
        s = np.divide(2*x[1], 1 - x[2]) - 1
        t = 2*x[2] - 1
        return (r, s, t)
    
    def diff_coordinate_transformation(self, x, i):
        if i == 0:
            return (np.divide(2, 1-x[1]-x[2]), 0, 0)
        elif i == 1:
            return (np.divide(2*x[0], (1-x[1]-x[2])**2), np.divide(2, 1-x[2]), 0)
        elif i == 2:
            return (np.divide(2*x[0], (1-x[1]-x[2])**2), np.divide(2*x[1], (1-x[2])**2), 2)
        else:
            raise Exception("Can't take the derivative in the {}th direction".format(i))

    def unroll_index(self, i):
        n = i[0]+i[1]+j[2]
        tet = (n * (n+1) * (n+2)) / 6.
        p = j[2]*(n+1) - j[2]*(j[2]-1)/2
        return int(tet + i[1] + p)
    
    def roll_index(self, x):
        # find the biggest tetrahedral number smaller than x
        # analytic inversion is horrible
        n = 0
        while((n+1) * (n+2) * (n+3) < 6*(x+1)):
            n+=1
        tet = (n * (n+1) * (n+2)) / 6.
    
        k = 0
        while((k+1)*(n+1) - k*(k+1)/2 < x + 1 - tet):
            k+=1
        p = k*(n+1) - k*(k-1)/2
    
        j = int(x - tet - p)
        i = int(n - j - k)
        return (i, j, k)

    def eval_basis(self, x, i):
        j = self.roll_index(i)
        r, s, t = self.coordinate_transformation(x)
        return theta_a(r, j[0]) * theta_b(s, j[0], j[1]) * theta_c(t, j[0], j[1], j[2])

    def eval_diff_basis(self, x, i, k):
        j = self.roll_index(i)
        r, s, t = self.coordinate_transformation(x)
        diff_r, diff_s, diff_t = self.diff_coordinate_transformation(x, k)
        return diff_theta_a(r, j[0]) * diff_r * theta_b(s, j[0], j[1]) * theta_c(t, j[0], j[1], j[2]) + \
                theta_a(r, j[0]) * diff_theta_b(s, j[0], j[1]) * diff_s * theta_c(t, j[0], j[1], j[2]) + \
                theta_a(r, j[0]) * theta_b(s, j[0], j[1]) * diff_theta_c(t, j[0], j[1], j[2]) * diff_t

    def number_of_basis_functions(self):
        return self.order * (self.order+1) * (self.order+2) // 6
