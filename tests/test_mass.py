#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests

from seissol_matrices import dg_matrices
from seissol_matrices import json_io
import helper


class abstract_tester(object):
    def test_M2(self):
        M2 = self.generator.face_generator.mass_matrix()
        M2_from_file = json_io.read_matrix(f"M2", self.filename)
        self.compare_matrices(M2, M2_from_file)

    def test_M3(self):
        M3 = self.generator.mass_matrix()
        M3_from_file = json_io.read_matrix(f"M3", self.filename)
        self.compare_matrices(M3, M3_from_file)

    def test_M2Inv(self):
        M2 = self.generator.face_generator.mass_matrix()
        M2inv = np.linalg.solve(M2, np.eye(M2.shape[0]))
        M2inv_from_file = json_io.read_matrix(f"M2inv", self.filename)
        self.compare_matrices(M2inv, M2inv_from_file)

    def test_M3Inv(self):
        M3 = self.generator.mass_matrix()
        M3inv = np.linalg.solve(M3, np.eye(M3.shape[0]))
        M3inv_from_file = json_io.read_matrix(f"M3inv", self.filename)
        self.compare_matrices(M3inv, M3inv_from_file)


def setUpClassFromOrder(cls, order):
    number_of_basis_functions = order * (order + 1) * (order + 2) // 6
    cls.filename = f"mass_{order}.json"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/v1.0.1/generated_code/matrices/mass_{order}.json"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    cls.generator = dg_matrices.dg_generator(cls.order, 3)


class test_stp_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 2)


class test_stp_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 3)


class test_stp_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 4)


class test_stp_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 5)


class test_stp_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 6)


class test_stp_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 7)


if __name__ == "__main__":
    unittest.main()
