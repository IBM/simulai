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

import torch

# SIREN (Sinusoidal Representation Networks)
class Siren(torch.nn.Module):

    name = 'Siren'

    def __init__(self, omega_0:float=None, c:float=None) -> None:

        super(Siren, self).__init__()

        self.omega_0 = omega_0
        self.c = c

    @property
    def share_to_host(self) -> dict:

        return {'omega_0': self.omega_0,
                'c': self.c}

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return torch.sin(self.omega_0*input)

# Sine activation
class sin(torch.nn.Module):

    name = 'sin'

    def __init__(self) -> None:

        super(sin, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:

        return torch.sin(input)
