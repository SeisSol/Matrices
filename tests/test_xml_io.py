#!/usr/bin/env python3

import unittest
import numpy as np
import os

from seissol_matrices import xml_io


class test_xml_io(unittest.TestCase):
    def test_write_read_one_matrix(self):
        filename = "random.xml"
        original_name = "random"
        original_matrix = np.random.rand(5, 5)
        xml_io.write_matrix(original_matrix, original_name, filename)
        read_matrix = xml_io.read_matrix(original_name, filename)
        for i in range(original_matrix.shape[0]):
            for j in range(original_matrix.shape[1]):
                self.assertAlmostEqual(original_matrix[i, j], read_matrix[i, j])
        os.remove(filename)

    def test_write_read_several_matrices(self):
        filename = "random.xml"
        original_names = ["random", "two", "three"]
        original_matrix = [
            np.random.rand(5, 5),
            np.random.rand(2, 3),
            np.random.rand(3, 2),
        ]
        for matrix, name in zip(original_matrix, original_names):
            xml_io.write_matrix(matrix, name, filename)

        for matrix, name in zip(original_matrix, original_names):
            read_matrix = xml_io.read_matrix(name, filename)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    self.assertAlmostEqual(matrix[i, j], read_matrix[i, j])
        os.remove(filename)
