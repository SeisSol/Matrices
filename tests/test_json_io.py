#!/usr/bin/env python3

import unittest
import numpy as np
import os

from seissol_matrices import json_io


class test_json_io(unittest.TestCase):
    def test_write_read_one_matrix(self):
        filename = "random.json"
        original_name = "random"
        original_matrix = np.random.rand(5, 5)
        json_io.write_matrix(original_matrix, original_name, filename)
        read_matrix = json_io.read_matrix(original_name, filename)
        for i in range(original_matrix.shape[0]):
            for j in range(original_matrix.shape[1]):
                self.assertAlmostEqual(original_matrix[i, j], read_matrix[i, j])
        os.remove(filename)

    def test_write_read_several_matrices(self):
        filename = "random.json"
        original_names = ["random", "two", "three"]
        original_matrix = [
            np.random.rand(5, 5),
            np.random.rand(2, 3),
            np.random.rand(3, 2),
        ]
        for matrix, name in zip(original_matrix, original_names):
            json_io.write_matrix(matrix, name, filename)

        for matrix, name in zip(original_matrix, original_names):
            read_matrix = json_io.read_matrix(name, filename)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    self.assertAlmostEqual(matrix[i, j], read_matrix[i, j])
        os.remove(filename)

    def test_read_one_matrix(self):
        filename = "tests/test.json"
        original_name = "test"
        original_matrix = np.array([[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        read_matrix = json_io.read_matrix(original_name, filename)
        for i in range(original_matrix.shape[0]):
            for j in range(original_matrix.shape[1]):
                self.assertAlmostEqual(original_matrix[i, j], read_matrix[i, j])
