from unittest import TestCase

import numpy as np


class TestKansas(TestCase):
    def setUp(self) -> None:
        pass

    def test_rbf_basis(self):
        from simulai.math.kansas import Kansas

        n_rbfs = 50
        n_series = n_rbfs

        data = np.random.rand(1_00, n_series)
        data_ext = np.random.rand(1_000, n_series)

        centroids = np.random.rand(n_series, n_rbfs)

        kansas = Kansas(centers=centroids, points=data)
