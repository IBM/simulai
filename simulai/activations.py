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


class Siren(torch.nn.Module):
    """Sinusoidal Representation Networks (SIREN)

    Parameters
    ----------
    omega_0 : float, optional
        Parameter for the SIREN model.
    c : float, optional
        Parameter for the SIREN model.

    """

    name = "Siren"

    def __init__(self, omega_0: float = None, c: float = None) -> None:
        """Initialize SIREN model with given parameters."""

        super(Siren, self).__init__()

        self.omega_0 = omega_0
        self.c = c

    @property
    def share_to_host(self) -> dict:
        """Return the parameters of the SIREN model.

        Returns
        -------
        params : dict
            A dictionary containing the parameters 'omega_0' and 'c'.

        """
        return {"omega_0": self.omega_0, "c": self.c}

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the SIREN model on the input.

        Parameters
        ----------
        input : torch.Tensor
            The input to the SIREN model.

        Returns
        -------
        output : torch.Tensor
            The output of the SIREN model.

        """
        return torch.sin(self.omega_0 * input)


class sin(torch.nn.Module):
    """Sine activation function.

    This module applies the sine function element-wise to the input.

    """

    name = "sin"

    def __init__(self) -> None:
        """Initialize the sine activation function."""

        super(sin, self).__init__()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Perform the forward pass of the sine activation function on the input.

        Parameters
        ----------
        input : torch.Tensor
            The input to the sine activation function.

        Returns
        -------
        output : torch.Tensor
            The output of the sine activation function.

        """
        return torch.sin(input)
