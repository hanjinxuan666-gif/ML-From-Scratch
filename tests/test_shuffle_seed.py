import unittest

import numpy as np

from mlfromscratch.utils.data_manipulation import train_test_split


class TrainTestSplitTests(unittest.TestCase):
    def test_zero_seed_reproduces_split(self):
        X = np.arange(30).reshape(15, 2)
        y = np.arange(15)
        state = np.random.get_state()
        try:
            np.random.seed(17)
            first = train_test_split(X, y, seed=0)
            np.random.seed(42)
            second = train_test_split(X, y, seed=0)
        finally:
            np.random.set_state(state)
        for left, right in zip(first, second):
            np.testing.assert_array_equal(left, right)
