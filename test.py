#!/usr/bin/env python3

import unittest
import numpy as np

import basis_functions

class TestBasisFunctions(unittest.TestCase):

    def test_1d(self):
        expected_results= np.load("test/basis_1d.npy")
        for order in range(8):
            for i in range(9):
                res = basis_functions.eval_basis_1d(0.1*(i+1), order)
                self.assertRelativelyEqual(expected_results[order, i], res)
    
    def test_2d(self):
        expected_results = np.load("test/basis_2d.npy")
        for number in range(54):
            for i in range(9):
                for j in range(9):
                    res = basis_functions.eval_basis_2d([0.1*(i+1), 0.1*(j+1)], basis_functions.number_to_index_2d(number))
                    self.assertRelativelyEqual(expected_results[number, i, j], res)

    def test_3d(self):
        expected_results = np.load("test/basis_3d.npy")
        for number in range(50):
            for i in range(9):
                for j in range(9):
                    for k in range(9):
                        if j + k != 8: 
                            res = basis_functions.eval_basis_3d([0.1*(i+1), 0.1*(j+1), 0.1*(k+1)], basis_functions.number_to_index_3d(number))
                            self.assertRelativelyEqual(expected_results[number, i, j, k], res)

    def test_diff_3d(self):
        for number in range(5):
            for i in range(3):
                for j in range(3):
                    for k in range(3):
                        if j+k != 8:
                            x = [0.1*(i+1), 0.1*(j+1), 0.1*(k+1)]
                            f = lambda xi: basis_functions.eval_basis_3d(xi, basis_functions.number_to_index_3d(number))
                            for l in range(3):
                                self.assertRelativelyEqual(self.numerical_gradient(f, x, l),\
                                basis_functions.eval_diff_basis_3d(x, basis_functions.number_to_index_3d(number), l), atol=1e-4, rtol=1e-4)
                            

    def test_index(self):
        self.assertEqual(basis_functions.index_2d_to_number(0, 0), 0)
        self.assertEqual(basis_functions.index_2d_to_number(1, 0), 1)
        self.assertEqual(basis_functions.index_2d_to_number(0, 1), 2)
        self.assertEqual(basis_functions.index_2d_to_number(2, 0), 3)
        self.assertEqual(basis_functions.index_2d_to_number(1, 1), 4)
        self.assertEqual(basis_functions.index_2d_to_number(0, 2), 5)
        self.assertEqual(basis_functions.index_2d_to_number(3, 0), 6)
        self.assertEqual(basis_functions.index_3d_to_number(0, 0, 0), 0)
        self.assertEqual(basis_functions.index_3d_to_number(1, 0, 0), 1)
        self.assertEqual(basis_functions.index_3d_to_number(0, 1, 0), 2)
        self.assertEqual(basis_functions.index_3d_to_number(0, 0, 1), 3)
        self.assertEqual(basis_functions.index_3d_to_number(2, 0, 0), 4)
        self.assertEqual(basis_functions.index_3d_to_number(1, 1, 0), 5)
        self.assertEqual(basis_functions.index_3d_to_number(0, 2, 0), 6)
        self.assertEqual(basis_functions.index_3d_to_number(1, 0, 1), 7)
        self.assertEqual(basis_functions.index_3d_to_number(0, 1, 1), 8)
        self.assertEqual(basis_functions.index_3d_to_number(0, 0, 2), 9)
        self.assertEqual(basis_functions.index_3d_to_number(3, 0, 0), 10)
        self.assertEqual(basis_functions.index_3d_to_number(2, 1, 0), 11)
        self.assertEqual(basis_functions.index_3d_to_number(1, 2, 0), 12)
        self.assertEqual(basis_functions.index_3d_to_number(0, 3, 0), 13)
        self.assertEqual(basis_functions.index_3d_to_number(2, 0, 1), 14)
        self.assertEqual(basis_functions.index_3d_to_number(1, 1, 1), 15)
        self.assertEqual(basis_functions.index_3d_to_number(0, 2, 1), 16)
        self.assertEqual(basis_functions.index_3d_to_number(1, 0, 2), 17)
        self.assertEqual(basis_functions.index_3d_to_number(0, 1, 2), 18)
        self.assertEqual(basis_functions.index_3d_to_number(0, 0, 3), 19)

    def test_thetha_a_derivative(self):
        for order in range(7):
            for i in range(9):
                f = lambda xi: basis_functions.theta_a(xi, order)
                self.assertRelativelyEqual(self.numerical_derivative(f, 0.1*(i+1)),\
                        basis_functions.diff_theta_a(0.1*(i+1), order), atol=1e-4, rtol=1e-4)

    def test_thetha_b_derivative(self):
        for order_i in range(7):
            for order_j in range(7):
                for i in range(9):
                    f = lambda xi: basis_functions.theta_b(xi, order_i, order_j)
                    self.assertRelativelyEqual(self.numerical_derivative(f, 0.1*(i+1)),\
                            basis_functions.diff_theta_b(0.1*(i+1), order_i, order_j), atol=1e-4, rtol=1e-4)

    def test_theta_c_derivative(self):
        for order_i in range(7):
            for order_j in range(7):
                for order_k in range(7):
                    for i in range(9):
                        f = lambda xi: basis_functions.theta_c(xi, order_i, order_j, order_k)
                        self.assertRelativelyEqual(self.numerical_derivative(f, 0.1*(i+1)),\
                                basis_functions.diff_theta_c(0.1*(i+1), order_i, order_j, order_k), atol=1e-4, rtol=1e-4)

    
    def numerical_derivative(self, f, x):
        h = 1e-11
        return (f(x+h) - f(x-h) ) / (2*h)

    def numerical_gradient(self, f, x, k):
        h = 1e-11
        basis = np.zeros(3)
        basis[k] = 1
        return (f(x+h*basis) - f(x-h*basis) ) / (2*h)

    def assertRelativelyEqual(self, a, b, atol=1e-10, rtol=1e-10):
        if not np.abs(a-b) < atol + rtol * np.abs(b):
            raise AssertionError("{} and {} are not relatively equal. atol = {}, rtol = {}".format(\
                    a, b, atol, rtol))

if __name__ == '__main__':
    unittest.main()



