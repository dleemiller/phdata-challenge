import unittest
import numpy as np

from preprocessing.scaler_999 import Scaler999


class TestScalers(unittest.TestCase):

    def test_scaler_999(self):
        test_data = np.array([1, 2, 999])
        s = Scaler999()
        s.fit(test_data)
        out_data = s.transform(test_data)

        # expect false for non-999 values
        self.assertListEqual([False, False, True], list(out_data))
