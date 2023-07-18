#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests

from seissol_matrices import stp_matrices
from seissol_matrices import json_io
import helper


class abstract_tester(object):
    def test_Z(self):
        Z = self.generator.Z()
        Z_from_file = json_io.read_matrix(f"Z", self.filename)
        self.compare_matrices(Z, Z_from_file)

    def test_wHat(self):
        wHat = self.generator.w()
        wHat = np.reshape(wHat, (wHat.shape[0], 1))
        wHat_from_file = json_io.read_matrix(f"wHat", self.filename)
        self.compare_matrices(wHat, wHat_from_file)

    def test_Zinv(self):
        Z = self.generator.Z()
        Zinv = np.linalg.solve(Z, np.eye(Z.shape[0]))

        Zinv_from_file = json_io.read_matrix(f"Zinv", self.filename)
        self.compare_matrices(Zinv, Zinv_from_file)

    def test_timeInt(self):
        timeInt = self.generator.time_int()
        timeInt = np.reshape(timeInt, (timeInt.shape[0], 1))
        timeInt_from_file = json_io.read_matrix(f"timeInt", self.filename)
        self.compare_matrices(timeInt, timeInt_from_file)

def setUpClassFromOrder(cls, order):
    number_of_basis_functions = order * (order + 1) * (order + 2) // 6
    cls.filename = f"stp_{order}.json"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/v1.0.1/generated_code/matrices/stp_{order}.json"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    cls.generator = stp_matrices.stp_generator(cls.order)


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

