#!/usr/bin/env python3

import unittest
import numpy as np
import os
import requests

from seissol_matrices import plasticity
from seissol_matrices import xml_io
from seissol_matrices import quad_points
from tests import helper


class abstract_tester(object):
    def test_nb(self):
        vandermonde = self.generator.generate_Vandermonde("nb")
        vandermonde_from_file = xml_io.read_matrix("v", self.filename)
        self.compare_matrices(vandermonde, vandermonde_from_file)

        vandermonde_inv = self.generator.generate_Vandermonde_inv("nb")
        vandermonde_inv_from_file = xml_io.read_matrix("vInv", self.filename)
        self.compare_matrices(vandermonde_inv, vandermonde_inv_from_file)


def setUpClassFromOrder(cls, order):
    cls.filename = f"plasticity_nb_{order}.xml"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/c99ca8f814b7184eb435272f9d4f63b03b8b6cf4/generated_code/matrices/plasticity_nb_matrices_{order}.xml"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    cls.generator = plasticity.PlasticityGenerator(cls.order)


class test_plasticity_2(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 2)


class test_plasticity_3(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 3)


class test_plasticity_4(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 4)


class test_plasticity_5(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 5)


class test_plasticity_6(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 6)


class test_plasticity_7(abstract_tester, helper.helper):
    @classmethod
    def setUpClass(cls):
        setUpClassFromOrder(cls, 7)


if __name__ == "__main__":
    unittest.main()
