from unittest import TestCase
import torch

from simulai.metrics import MahalanobisDistance

class TestMahalanobis(TestCase):

    def test_mahalanobis(self):

        data = torch.randn(100, 10)

        center = torch.randn(100, 10)

        metric_tensor = torch.eye(10)

        metric = MahalanobisDistance(batchwise=True)

        value = metric(metric_tensor=metric_tensor, center=center, point=data)

        metric = MahalanobisDistance()

        value = metric(metric_tensor=metric_tensor, center=center[0], point=data[0])

        print(value)
