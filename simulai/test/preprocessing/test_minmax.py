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

from simulai.metrics import MinMaxEvaluation

class TestMinMax(TestCase):

    def setUp(self) -> None:

        pass

    def test_minmax(self):

        arr = np.random.rand(1_000, 10, 10, 3)

        minmax = MinMaxEvaluation()

        minmax(dataset=arr, data_interval=[0, arr.shape[0]], batch_size=1_000, axis=3)

