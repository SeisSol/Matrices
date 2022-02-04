#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests

from seissol_matrices import dr_matrices
from seissol_matrices import json_io
from seissol_matrices import quadrature
from tests import helper


class abstract_tester(object):
    def test_V3mTo2n(self):
        for a in range(0, 4):
            for b in range(0, 4):
                V3mTo2n = self.generator.V3mTo2n(a, b)
                V3mTo2n_from_file = json_io.read_matrix(
                    f"V3mTo2n({a},{b})", self.filename
                )
                self.compare_matrices(V3mTo2n, V3mTo2n_from_file)

    def test_V3mTo2nTWDivM(self):
        np.set_printoptions(linewidth=200)
        for a in range(4):
            for b in range(4):
                V3mTo2nTWDivM = self.generator.V3mTo2nTWDivM(a, b)
                V3mTo2nTWDivM_from_file = json_io.read_matrix(
                    f"V3mTo2nTWDivM({a},{b})", self.filename
                )
                self.compare_matrices(V3mTo2nTWDivM, V3mTo2nTWDivM_from_file)


def setUpClassFromOrder(cls, order):
    cls.filename = f"dr_quadrature_matrices_{order}.json"
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/master/generated_code/matrices/dr_quadrature_matrices_{order}.json"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    quadrule = quadrature.gauss_jacobi()
    cls.generator = dr_matrices.dr_generator(cls.order, quadrule)


class test_dr_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 2)


class test_dr_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 3)


class test_dr_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 4)


class test_dr_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 5)


class test_dr_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 6)


class test_dr_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 7)


if __name__ == "__main__":
    unittest.main()
