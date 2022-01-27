#!/usr/bin/env python3

import unittest
import numpy as np
import os

from seissol_matrices import json_io


class test_json_io(unittest.TestCase):
    def test_read_one_matrix(self):
        filename = "tests/test.json"
        original_name = "test"
        original_matrix = np.array([[2, 3, 4, 5], [3, 4, 5, 6], [4, 5, 6, 7]])
        read_matrix = json_io.read_matrix(original_name, filename)
        for i in range(original_matrix.shape[0]):
            for j in range(original_matrix.shape[1]):
                self.assertAlmostEqual(original_matrix[i, j], read_matrix[i, j])
