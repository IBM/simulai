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

