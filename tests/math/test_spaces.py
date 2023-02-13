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

from simulai.math.spaces import GaussianRandomFields


class TestSpaces(TestCase):
    def setUp(self) -> None:
        pass

    def test_gaussian_random_fields(self):
        n_features = 100
        M = 1_00
        n_sensors = 100
        N = 100

        grf = GaussianRandomFields(N=N)

        u = grf.random_u(n_features=n_features)

        assert isinstance(u, np.ndarray)

        features = np.random.rand(n_features, M)
        sensors = np.random.rand(n_sensors)

        u_bar = grf.generate_u(features=features, sensors=sensors)

        assert isinstance(u_bar, np.ndarray)
