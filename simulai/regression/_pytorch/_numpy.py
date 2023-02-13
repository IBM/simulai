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
import torch


class LinearNumpy:
    def __init__(self, layer: torch.nn.Linear = None, name: str = None) -> None:
        self.weights = layer.weight.detach().numpy()

        if layer.bias is not None:
            self.bias = layer.bias.detach().numpy()
        else:
            self.bias = 0

        self.name = name

    @property
    def weights_l2(self):
        return sum([np.linalg.norm(weight, 2) for weight in self.weights])

    @property
    def weights_l1(self):
        return sum([np.linalg.norm(weight, 1) for weight in self.weights])

    def forward(self, input_field: np.ndarray = None) -> np.ndarray:
        return input_field @ self.weights.T + self.bias

    def eval(self, input_field: np.ndarray = None) -> np.ndarray:
        return self.forward(input_field=input_field)
