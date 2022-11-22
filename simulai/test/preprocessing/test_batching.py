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
import random

from simulai.batching import batchdomain_constructor

class TestBatching(TestCase):

    def test_batching_data_interval(self):

        data_interval = [0, 100]
        batch_size = 10

        batches = batchdomain_constructor(data_interval=data_interval, batch_size=batch_size)

        assert isinstance(batches, list)

    def test_batching_batch_indices(self):

        n_samples = 100
        indices = sorted([random.randint(0, n_samples) for i in range(n_samples)])

        batch_size = 10

        batches = batchdomain_constructor(batch_indices=indices, batch_size=batch_size)

        assert isinstance(batches, list)
