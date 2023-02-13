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

from typing import Union

import numpy as np
import torch
from scipy.sparse.csc import csc_matrix

from simulai.math.kansas import Kansas
from simulai.templates import NetworkTemplate, as_array, as_tensor


# Single-layer RBF network
class RBFLayer(torch.nn.Module):
    def __init__(
        self,
        xmin: Union[float, np.ndarray] = None,
        xmax: Union[float, np.ndarray] = None,
        Nk: int = None,
        name: str = None,
        var_dim: int = None,
        Mu: torch.Tensor = None,
        Sigma: Union[float, torch.Tensor] = None,
        device: str = None,
    ) -> None:
        super(RBFLayer, self).__init__()

        if (
            isinstance(xmax, np.ndarray) == True
            and isinstance(xmin, np.ndarray) == True
        ):
            self.xmin = torch.Tensor(xmin).detach()
            self.xmax = torch.Tensor(xmax).detach()

        elif isinstance(xmax, int) == True and isinstance(xmin, int) == True:
            self.xmin = torch.Tensor([xmin]).detach()
            self.xmax = torch.Tensor([xmax]).detach()

        else:
            raise Exception(
                f"Both xmin and xmax must be numpy arrays or integers,"
                f" but received {xmax} and {xmin}."
            )

        self.space_dim = len(self.xmax)

        self.Nk = Nk
        self.name = name
        self.var_dim = var_dim
        self.output_size = var_dim * Nk
        self.device = device

        if Mu == None:
            self.Mu = (
                torch.rand((self.Nk)) * (self.xmax - self.xmin) + self.xmin
            ).detach()
        else:
            self.Mu = Mu

        if Sigma == None:
            self.Sigma = (
                torch.rand((self.Nk)) * (self.xmax - self.xmin) + self.xmin
            ).detach()
        else:
            if isinstance(Sigma, float):
                self.Sigma = torch.Tensor([Sigma])
            else:
                self.Sigma = Sigma

        self.W = torch.ones((self.Nk, 1))

        self.weights = [self.W]

        self.Mu = self.Mu.to(self.device)
        self.Sigma = self.Sigma.to(self.device)

    def set_W(self, W: torch.Tensor = None) -> None:
        setattr(self, "W", W)

    def set_Sigma(self, Sigma: torch.Tensor = None) -> None:
        setattr(self, "Sigma", Sigma)

    def set_Mu(self, Mu: torch.Tensor = None) -> None:
        setattr(self, "Mu", Mu)

    def basis(self, input_data: Union[torch.Tensor, np.ndarray] = None) -> torch.Tensor:
        exponent = torch.pow(input_data - self.Mu, 2) / self.Sigma

        rbf_interpolation = torch.exp(-exponent)

        return rbf_interpolation

    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        return self.basis(input_data=input_data)

    def eval(self, input_data: Union[np.ndarray, torch.Tensor] = None) -> np.ndarray:
        output_tensor = self.forward(input_data=input_data)

        # Guaranteeing the dataset location as CPU
        output_tensor = output_tensor.to("cpu")

        return output_tensor.detach().numpy()

    def summary(self):
        print(f"RBF layer with {self.Nk} basis.")


class ModalRBFNetwork(NetworkTemplate):
    def __init__(
        self,
        xmin: Union[float, np.ndarray] = None,
        xmax: Union[float, np.ndarray] = None,
        Nk: int = None,
        name: str = None,
        var_dim: int = None,
        Mu: Union[torch.Tensor, np.ndarray] = None,
        Sigma: Union[float, torch.Tensor] = None,
        coeff_network: NetworkTemplate = None,
    ) -> None:
        super(ModalRBFNetwork, self).__init__()

        self.Nk = Nk
        self.var_dim = var_dim

        self.rbf_basis = RBFLayer(
            xmin=xmin, xmax=xmax, Nk=Nk, name=name, var_dim=var_dim, Mu=Mu, Sigma=Sigma
        )

        self.coeff_network = coeff_network

        self.Mu = Mu
        self.Sigma = Sigma
        self.W = torch.ones((self.Nk, 1))

        self.weights = [self.W]

    @as_tensor
    def forward(
        self, input_data: Union[torch.Tensor, np.ndarray] = None
    ) -> torch.Tensor:
        space_input = input_data[:, :-1]
        time_input = input_data[:, -1:]

        output = self.coeff_network.forward(input_data=time_input)

        self.W = output

        space_component = self.rbf_basis.basis(input_data=space_input)

        space_component_reshaped = torch.reshape(
            space_component, (-1, self.var_dim, self.Nk)
        )

        output = torch.sum(
            space_component_reshaped * self.W[:, None, :], dim=-1, keepdim=False
        )

        output = torch.squeeze(output)

        return output

    def summary(self):
        self.coeff_network.summary()


class RBFNetwork(NetworkTemplate):
    def __init__(
        self,
        hidden_units: int = None,
        output_layer: bool = True,
        xmin: float = None,
        xmax: float = None,
        Nk: int = None,
        name: str = None,
        output_size: int = None,
    ):
        super(RBFNetwork, self).__init__()

        self.Nk = Nk
        self.output_size = output_size

        if hidden_units is None:
            self.input_layer = lambda x: x

        if output_layer is False:
            self.output_layer = lambda x: x
        else:
            self.output_layer = torch.nn.Linear(Nk, output_size)

        self.xk = np.linspace(xmin, xmax, self.Nk)[:, None]

        self.sigma2 = ((np.max(self.xk) - np.min(self.xk)) / np.sqrt(2 * self.Nk)) ** 2

        self.name = name

        self.layers = [self.output_layer]
        self.weights = [item.weight for item in self.layers]

    @as_tensor
    def _last_layer(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        return self.output_layer(input_data)

    @as_array
    def _forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> csc_matrix:
        hidden_state = self.input_layer(input_data)

        self.Kg = Kansas(hidden_state, self.xk, sigma2=self.sigma2)

        G = self.Kg.get_interpolation_matrix()

        return G

    @as_array
    def _gradient(self, input_data: Union[np.ndarray, torch.Tensor]) -> csc_matrix:
        hidden_state = self.input_layer(input_data)

        self.Kg = Kansas(hidden_state, self.xk, sigma2=self.sigma2)

        Dx = self.Kg.get_first_derivative_matrix(0)

        return Dx

    @as_tensor
    def forward(
        self, input_data: Union[np.ndarray, torch.Tensor] = None
    ) -> torch.Tensor:
        G = self._forward(input_data=input_data)

        activated_state = np.array(G.todense())

        return self._last_layer(input_data=activated_state)

    def gradient(
        self,
        ref_data: Union[np.ndarray, torch.Tensor],
        input_data: Union[np.ndarray, torch.Tensor] = None,
    ) -> torch.Tensor:
        Dx = self._gradient(input_data=input_data)

        activated_state = np.array(Dx.todense())

        return self._last_layer(input_data=activated_state)

    def fit(
        self,
        input_data: Union[np.ndarray, torch.Tensor] = None,
        target_data: Union[np.ndarray, torch.Tensor] = None,
    ):
        G = self._forward(input_data=input_data)

        self.weights = np.linalg.pinv(np.array(G.todense())) @ target_data

    def eval(self, input_data=None):
        G = self._forward(input_data=input_data)

        return G @ self.weights
