import numpy as np
from unittest import TestCase

from simulai.math.products import kronecker

class TestProducts(TestCase):

    def setUp(self) -> None:
        pass

    def test_kronecker(self):

        n_features = [5, 10, 20, 40]

        for n in n_features:

            a = np.random.rand(1_000, n)

            a_ = kronecker(a=a)

            n_extended = a_.shape[1]

            assert len(a_.shape) == 2
            assert n_extended == (n)*(n + 1)/2

