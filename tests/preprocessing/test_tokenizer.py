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

import random
from unittest import TestCase
import numpy as np

from simulai.io import Tokenizer

class TestTokenizer(TestCase):

    def test_time_indexer(self):

        n_samples = 100
        n_series_input = 3
        n_series_output = 2
        num_step = 5
        step = 1

        input_data = np.random.rand(n_samples, n_series_input)
        target_data = np.random.rand(n_samples, n_series_output)

        tokenizer = Tokenizer(kind="time_indexer")

        input_dataset = tokenizer.generate_input_tokens(input_data, num_step=num_step, step=step)
        target_dataset = tokenizer.generate_target_tokens(target_data, num_step=num_step)

        print(f"Input shape: {input_dataset.shape}")
        print(f"Target shape: {target_dataset.shape}")

    def test_time_example(self):

        n_samples = 10
        num_step = 5
        step = 0.1

        input_data = np.arange(0, 10, 1)[:, None]
        target_data = np.arange(10, 20, 1)[:, None]

        tokenizer = Tokenizer(kind="time_indexer")

        input_dataset = tokenizer.generate_input_tokens(input_data, num_step=num_step, step=step)
        target_dataset = tokenizer.generate_target_tokens(target_data, num_step=num_step)

        print(input_dataset)
        print(target_dataset)
