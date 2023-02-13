import sys
import warnings

import numpy as np
import torch
from torch.autograd import grad

warnings.warn("Hamilatonian sampling is in EXPERIMENTAL stage.")

from simulai.metrics import MahalanobisDistance
from simulai.templates import NetworkTemplate

"""This sampling technique is present in:

 Chadebec, Clément, and Stéphanie Allassonnière.
 "A Geometric Perspective on Variational Autoencoders."
 arXiv preprint arXiv:2209.07370 (2022).
 
"""


# Basis used for interpolating over the Riemaniann space
class Omega:
    def __init__(
        self,
        rho: float = None,
        mu: torch.Tensor = None,
        covariance: torch.Tensor = None,
        batchwise: bool = None,
    ):
        self.rho = rho
        self.mu = mu
        self.covariance = covariance
        self.batchwise = batchwise
        self.metric = MahalanobisDistance(batchwise=batchwise)

    def __call__(self, z: torch.Tensor = None):
        metric = torch.sqrt(
            self.metric(center=self.mu, metric_tensor=self.covariance, point=z)
        )

        return torch.exp(-(metric**2) / (self.rho**2))


# Riemaniann metric
class G_metric:
    def __init__(
        self,
        k: int = None,
        rho: float = None,
        tau: float = None,
        lambd: float = None,
        model: NetworkTemplate = None,
        input_data: torch.Tensor = None,
        batchwise: bool = None,
    ):
        self.k = k
        self.tau = tau
        self.lambd = lambd
        self.n_samples = input_data.shape[0]

        Mu = model.Mu(input_data=input_data)
        Covariance = model.CoVariance(input_data=input_data)

        indices = np.random.choice(self.n_samples, k)

        self.latent_dim = Mu.shape[1]

        if rho != None:
            self.rho = rho
        else:
            self.rho = self.rho_criterion(c=Mu[indices])

        self.basis = [
            Omega(rho=self.rho, mu=Mu[i], covariance=Covariance[i], batchwise=batchwise)
            for i in indices
        ]
        self.covariances = [Covariance[i] for i in indices]

        self.latent_dim = Mu.shape[1]

    def rho_criterion(self, c: torch.Tensor = None):
        differences = torch.stack(
            [
                c[:, j : j + 1] - c[:, j : j + 1].T * torch.ones(self.k, self.k)
                for j in range(self.latent_dim)
            ],
            dim=2,
        )

        differences_norm = torch.linalg.norm(differences, dim=-1) + 1e16 * torch.eye(
            self.k
        )

        min_dist_between_neigh = torch.min(differences_norm, dim=-1).values

        rho = torch.max(min_dist_between_neigh)

        return rho

    def G(self, z: torch.Tensor = None) -> torch.Tensor:
        if z.requires_grad == False:
            z.requires_grad = True

        g_z = sum(
            [cov * basis(z=z) for cov, basis in zip(self.covariances, self.basis)]
        ) + self.lambd * torch.exp(-self.tau * torch.linalg.norm(z, 2)) * torch.eye(
            self.latent_dim
        )

        return g_z

    def G_diag(self, z: torch.Tensor = None) -> torch.Tensor:
        return self.G(z=z).diag()

    def G_grad_z(self, z: torch.Tensor = None) -> torch.Tensor:
        G_diag = self.G_diag(z=z)

        G_diag_split = torch.split(G_diag, 1, dim=-1)

        g_grad_z = torch.hstack([self.gradient(gg, z)[:, None] for gg in G_diag_split])

        return g_grad_z

    def __call__(self, z: torch.Tensor = None):
        z.requires_grad = True

        g_z = self.G(z=z)

        G_z = torch.sqrt(torch.det(g_z))

        return G_z

    @staticmethod
    def gradient(feature, param):
        grad_ = grad(
            feature,
            param,
            grad_outputs=torch.ones_like(feature),
            create_graph=True,
            allow_unused=True,
            retain_graph=True,
        )

        return grad_[0]


# The Hamiltonian system we are interested in
class HamiltonianEquations:
    def __init__(self, metric: G_metric = None):
        self.metric = metric
        self.latent_dim = metric.latent_dim

    def H_til(self, z: torch.Tensor = None):
        return -(1 / 2) * torch.log(torch.det(self.metric.G(z=z)))

    def __call__(self, z: torch.Tensor = None, v: torch.Tensor = None):
        G_inv = self.metric.G(z=z).inverse()
        G_grad_z_value = self.metric.G_grad_z(z=z)

        dHdz = torch.hstack(
            [
                -0.5 * (G_inv @ (G_grad_z_value[j][None, :].T)).trace()
                for j in range(z.shape[0])
            ]
        )

        return dHdz


# Basic Leapfrog integrator
class LeapFrogIntegrator:
    def __init__(
        self, system: callable = None, n_steps: int = None, e_lf: float = None
    ):
        self.system = system
        self.latent_dim = self.system.latent_dim

        self.n_steps = n_steps
        self.e_lf = e_lf

        self.log_phrase = "LeapFrog Integration"

    def step(self, v: torch.Tensor = None, z: torch.Tensor = None):
        dHdz = self.system(z=z, v=v)

        v_bar = v - (self.e_lf / 2) * dHdz
        z_til = z + self.e_lf * v_bar

        dHdz_til = self.system(z=z_til, v=v_bar)

        v_til = v_bar - (self.e_lf / 2) * dHdz_til

        return z_til, v_til

    def solve(self, z_0: torch.Tensor = None, v_0: torch.Tensor = None):
        z = z_0
        v = v_0

        for k in range(self.n_steps):
            sys.stdout.write(
                "\r {}, iteration: {}/{}".format(self.log_phrase, k + 1, self.n_steps)
            )
            sys.stdout.flush()

            z_til, v_til = self.step(v=v, z=z)

            z = z_til
            v = v_til

        return z, v


# Hamiltonian sampling
class HMC:
    def __init__(self, integrator: LeapFrogIntegrator = None, N: int = int):
        self.integrator = integrator
        self.H_system = integrator.system
        self.latent_dim = integrator.latent_dim

        self.N = N

    def stopping_criterion(self, z: torch.Tensor = None, z_0: torch.Tensor = None):
        H_0 = self.H_system.H_til(z=z)
        H = self.H_system.H_til(z=z_0)

        alpha = torch.min(torch.Tensor([1.0]), torch.exp(H_0 - H))

        return alpha

    def solve(self, z_0: torch.Tensor = None):
        z = z_0

        for j in range(self.N):
            print(f"Iteration {j} of a chain with size {self.N}.")

            v = torch.randn(self.latent_dim)

            z_til, v_til = self.integrator.solve(z_0=z, v_0=v)

            alpha = self.stopping_criterion(z=z_til, z_0=z_0)

            z = z_0 * alpha + (1 - alpha) * z

        return z
