from unittest import TestCase

import numpy as np
import torch

from simulai.metrics import PointwiseError

class TestPointwise(TestCase):

    def test_pointwise(self):

        data = torch.randn(100, 10)

        data_ref = torch.randn(100, 10)

        metric = PointwiseError()

        evaluation = metric(data=data, reference_data=data_ref)

        assert isinstance(evaluation, np.ndarray)

        data = torch.randn(100, 10)
        data[10, 1] = np.NaN
        data[50, 2] = np.inf

        data_ref = torch.randn(100, 10)

        metric = PointwiseError()

        evaluation = metric(data=data, reference_data=data_ref)

        assert isinstance(evaluation, np.ndarray)
       
