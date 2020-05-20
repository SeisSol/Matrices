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

def coordinate_transformation_1d(x):
    return 2*x - 1

def coordinate_transformation_2d(x, y):
    r = np.divide(2*x, 1-y) - 1
    s = 2*y -1
    return (r, s)

def coordinate_transformation_3d(x, y, z):
    r = np.divide(2*x, 1-y-z) - 1
    s = np.divide(2*y, 1 - z) - 1
    t = 2*z - 1
    return (r, s, t)

def diff_coordinate_transformation_3d(x, y, z, i):
    if i == 0:
        return (np.divide(2, 1-y-z), 0, 0)
    elif i == 1:
        return (np.divide(2*x, (1-y-z)**2), np.divide(2, 1-z), 0)
    elif i == 2:
        return (np.divide(2*x, (1-y-z)**2), np.divide(2*y, (1-z)**2), 2)
    else:
        raise Exception("Can't take the derivative in the {}th direction".format(i))

def eval_basis_1d(x, i):
    r = coordinate_transformation_1d(x)
    return theta_a(r, i)

def eval_basis_2d(x, i):
    r, s = coordinate_transformation_2d(x[0], x[1])
    return theta_a(r, i[0]) * theta_b(s, i[0], i[1])

def eval_basis_3d(x, i):
    r, s, t = coordinate_transformation_3d(x[0], x[1], x[2])
    return theta_a(r, i[0]) * theta_b(s, i[0], i[1]) * theta_c(t, i[0], i[1], i[2])

#computes d / d_x_k phi_i (x)
def eval_diff_basis_3d(x, i, k):
    r, s, t = coordinate_transformation_3d(x[0], x[1], x[2])
    diff_r, diff_s, diff_t = diff_coordinate_transformation_3d(x[0], x[1], x[2], k)
    return diff_theta_a(r, i[0]) * diff_r * theta_b(s, i[0], i[1]) * theta_c(t, i[0], i[1], i[2]) + \
            theta_a(r, i[0]) * diff_theta_b(s, i[0], i[1]) * diff_s * theta_c(t, i[0], i[1], i[2]) + \
            theta_a(r, i[0]) * theta_b(s, i[0], i[1]) * diff_theta_c(t, i[0], i[1], i[2]) * diff_t

def index_2d_to_number(i, j):
    n = i+j
    tri = 0.5 * n * (n+1)
    return int(tri + j)

def number_to_index_2d(x):
    n = int(-0.5+np.sqrt(0.25 + 2*x))
    tri = int(0.5 * n * (n+1))
    j = int(x - tri)
    i = int(n - j)
    return(i, j)

def index_3d_to_number(i, j, k):
    n = i+j+k
    tet = (n * (n+1) * (n+2)) / 6.
    p = k*(n+1) - k*(k-1)/2
    return int(tet + j + p)

def number_to_index_3d(x):
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

def number_of_3d_basis_functions(order):
    return int(order * (order+1) * (order+2) / 6)
