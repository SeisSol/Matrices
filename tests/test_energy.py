#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests

from seissol_matrices import energy, quad_points
from seissol_matrices import xml_io
from tests import helper


class abstract_tester(object):
    def test_V2mTo2JacobiQuad(self):
        quadrule = quad_points.stroud(self.order + 1)
        V2mTo2 = self.generator.generate_V2mTo2n(order=self.order, quadrule=quadrule)
        V2mTo2_from_file = xml_io.read_matrix(f"V2mTo2JacobiQuad", self.filename)
        self.compare_matrices(V2mTo2, V2mTo2_from_file)


def setUpClassFromOrder(cls, order):
    number_of_basis_functions = order * (order + 1) * (order + 2) // 6
    cls.filename = f"gravitational_energy_matrices_{order}.xml"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/v1.0.1/generated_code/matrices/nodal/gravitational_energy_matrices_{order}.xml"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    cls.generator = energy


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
