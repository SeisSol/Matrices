#!/usr/bin/env python3

import unittest
import numpy as np
import os

from seissol_matrices import xml_io

class TestXMLIO(unittest.TestCase):

    def test_write_read(self):
        filename = "random.xml"
        original_name = "random"
        original_matrix = np.random.rand(5, 5)
        xml_io.write_matrix(original_matrix, original_name, filename)
        read_matrix, read_name = xml_io.read_matrix(filename)
        self.assertEqual(original_name, read_name)
        for i in range(original_matrix.shape[0]): 
            for j in range(original_matrix.shape[1]): 
                self.assertAlmostEqual(original_matrix[i, j], read_matrix[i, j])
        os.remove(filename)


        
