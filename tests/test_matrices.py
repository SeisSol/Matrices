#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests

from seissol_matrices import dg_matrices
from seissol_matrices import xml_io
import helper


class abstract_tester(object):
    def test_kDivMT(self):
        for i in range(3):
            kDivMT = self.generator.kDivMT(i)
            kDivMT_from_file = xml_io.read_matrix(f"kDivMT({i})", self.filename)
            self.compare_matrices(kDivMT, kDivMT_from_file)

    def test_kDivM(self):
        for i in range(3):
            kDivM = self.generator.kDivM(i)
            kDivM_from_file = xml_io.read_matrix(f"kDivM({i})", self.filename)
            self.compare_matrices(kDivM, kDivM_from_file)

    def test_fMrT(self):
        for i in range(4):
            fMrT = self.generator.fMrT(i)
            fMrT_from_file = xml_io.read_matrix(f"fMrT({i})", self.filename)
            self.compare_matrices(fMrT, fMrT_from_file)

    def test_rT(self):
        for i in range(4):
            rT = self.generator.rT(i)
            rT_from_file = xml_io.read_matrix(f"rT({i})", self.filename)
            self.compare_matrices(rT, rT_from_file)

    def test_rDivM(self):
        for i in range(4):
            rDivM = self.generator.rDivM(i)
            rDivM_from_file = xml_io.read_matrix(f"rDivM({i})", self.filename)
            self.compare_matrices(rDivM, rDivM_from_file)

    def test_fP(self):
        for i in range(3):
            fP = self.generator.fP(i)
            fP_from_file = xml_io.read_matrix(f"fP({i})", self.filename)
            self.compare_matrices(fP, fP_from_file)


def setUpClassFromOrder(cls, order):
    number_of_basis_functions = order * (order + 1) * (order + 2) // 6
    cls.filename = f"matrices_{number_of_basis_functions}.xml"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/c99ca8f814b7184eb435272f9d4f63b03b8b6cf4/generated_code/matrices/matrices_{number_of_basis_functions}.xml"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    cls.generator = dg_matrices.dg_generator(cls.order, 3)


class test_matrices_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 2)


class test_matrices_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 3)


class test_matrices_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 4)


class test_matrices_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 5)


class test_matrices_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 6)


class test_matrices_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 7)


if __name__ == "__main__":
    unittest.main()
