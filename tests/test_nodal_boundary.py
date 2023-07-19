#!/usr/bin/env python3

import unittest
import requests

from seissol_matrices import nodal_boundary, json_io
import helper


class abstract_tester(object):
    def test_nodes(self):
        nodes = self.generator.nodes
        nodes_from_file = json_io.read_matrix("nodes2D", self.filename)
        self.compare_matrices(nodes, nodes_from_file)

    def test_V2mTo2n(self):
        V2mTo2n = self.generator.generate_V2mTo2n()
        V2mTo2n_from_file = json_io.read_matrix("V2mTo2n", self.filename)
        self.compare_matrices(V2mTo2n, V2mTo2n_from_file)

    def test_V2nTo2m(self):
        V2nTo2m = self.generator.generate_V2nTo2m()
        V2nTo2m_from_file = json_io.read_matrix("V2nTo2m", self.filename)
        self.compare_matrices(V2nTo2m, V2nTo2m_from_file)

    def test_MV2nTo2m(self):
        MV2nTo2m = self.generator.generate_MV2nTo2m()
        MV2nTo2m_from_file = json_io.read_matrix("MV2nTo2m", self.filename)
        self.compare_matrices(MV2nTo2m, MV2nTo2m_from_file)

    def test_V3mTo2nFace(self):
        for faceid in range(4):
            V3mTo2nFace = self.generator.generate_V3mTo2nFace(faceid)
            V3mTo2nFace_from_file = json_io.read_matrix(
                f"V3mTo2nFace({faceid})", self.filename
            )
            self.compare_matrices(V3mTo2nFace, V3mTo2nFace_from_file)


def setUpClassFromOrder(cls, order):
    cls.filename = f"nodalBoundary_matrices_{order}.json"
    # point to some old commit
    url = f"https://raw.githubusercontent.com/SeisSol/SeisSol/c99ca8f814b7184eb435272f9d4f63b03b8b6cf4/generated_code/matrices/nodal/nodalBoundary_matrices_{order}.json"
    r = requests.get(url, allow_redirects=True)
    open(cls.filename, "wb").write(r.content)
    cls.order = order
    cls.generator = nodal_boundary.NodalBoundaryGenerator(order)


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
