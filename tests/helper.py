import unittest
import os


class helper(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        os.remove(cls.filename)

    def compare_matrices(self, a, b):
        self.assertEqual(a.shape, b.shape)
        for i in range(a.shape[0]):
            for j in range(b.shape[1]):
                self.assertAlmostEqual(a[i, j], b[i, j])
