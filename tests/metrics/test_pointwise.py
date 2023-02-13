# (C) Copyright IBM Corp. 2019, 2020, 2021, 2022.

#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at

#           http://www.apache.org/licenses/LICENSE-2.0

#     Unless required by applicable law or agreed to in writing, software
#     distributed under the License is distributed on an "AS IS" BASIS,
#     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#     See the License for the specific language governing permissions and
#     limitations under the License.

from unittest import TestCase

import numpy as np
import torch

from simulai.metrics import PointwiseError


class TestPointwise(TestCase):
    def test_pointwise(self):
        data = np.random.rand(100, 10)

        data_ref = np.random.rand(100, 10)

        metric = PointwiseError()

        evaluation = metric(data=data, reference_data=data_ref)

        assert isinstance(evaluation, np.ndarray)

        data = np.random.rand(100, 10)
        data[10, 1] = np.NaN
        data[50, 2] = np.inf

        data_ref = np.random.rand(100, 10)

        metric = PointwiseError()

        evaluation = metric(data=data, reference_data=data_ref)

        assert isinstance(evaluation, np.ndarray)
