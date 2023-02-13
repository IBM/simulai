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

# Written by Imran Nasim adapted from https://github.com/gbdl/BBI

import torch
from torch.optim.optimizer import Optimizer  # , required


class BBI(Optimizer):
    """Optimizer based on the BBI model of inflation.

    Args:
    params (iterable): iterable of parameters to optimize or dicts defining parameter groups
    lr (float): learning rate
    v0 (float): expected minimum of the potential (Delta V in the paper)
    threshold0 (int): threshold for fixed bounces (T0 in the paper)
    threshold1 (int): threshold for progress-dependent bounces (T1 in the paper)
    deltaEn (float): extra initial energy (delta E in the paper)
    consEn (bool): if True enforces energy conservation at every step
    n_fixed_bounces (int): number of bounces every T0 iterations (Nb in the paper)
    """

    def __init__(
        self,
        params: dict = None,
        lr: float = 1e-3,
        eps1: float = 1e-10,
        eps2: float = 1e-40,
        v0: float = 0,
        threshold0: int = 1000,
        threshold: int = 3000,
        deltaEn: float = 0.0,
        consEn: bool = True,
        n_fixed_bounces: int = 1,
    ) -> None:
        defaults = dict(
            lr=lr,
            eps1=eps1,
            eps2=eps2,
            v0=v0,
            threshold=threshold,
            threshold0=threshold0,
            deltaEn=deltaEn,
            consEn=consEn,
            n_fixed_bounces=n_fixed_bounces,
        )
        self.energy = None
        self.min_loss = None
        self.iteration = 0
        self.deltaEn = deltaEn
        self.n_fixed_bounces = n_fixed_bounces
        self.consEn = consEn
        super(BBI, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(BBI, self).__setstate__(state)

    def step(self, closure: callable) -> torch.Tensor:
        loss = closure()  # .item()

        # initialization
        if self.iteration == 0:
            # define a random numbers generator, in order not to use the ambient seed and have random bounces even with the same ambient seed
            self.generator = torch.Generator(
                device=self.param_groups[0]["params"][0].device
            )
            self.generator.manual_seed(self.generator.seed() + 1)

            # Initial energy
            self.initV = loss - self.param_groups[0]["v0"]
            self.init_energy = self.initV + self.deltaEn

            # Some counters
            self.counter0 = 0
            self.fixed_bounces_performed = 0
            self.counter = 0

            self.min_loss = float("inf")

        for group in self.param_groups:
            V = loss - group["v0"]
            dt = group["lr"]
            eps1 = group["eps1"]
            eps2 = group["eps2"]
            threshold0 = group["threshold0"]
            threshold = group["threshold"]

            if V > eps2:
                EoverV = self.init_energy / V
                VoverE = V / self.init_energy

                # Now I check if loss and pi^2 are consistent with the initial value of the energy

                ps2_pre = torch.tensor(
                    0.0, device=self.param_groups[0]["params"][0].device
                )

                for p in group["params"]:
                    param_state = self.state[p]
                    d_p = p.grad.data
                    # Initialize in the direction of the gradient, with magnitude related to deltaE
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = -(
                            d_p / torch.norm(d_p)
                        ) * torch.sqrt(
                            torch.tensor(
                                ((self.init_energy**2) / self.initV) - self.initV
                            )
                        )
                    else:
                        buf = param_state["momentum_buffer"]

                    # compute the current pi^2 . Pre means that this is the value before the iteration step
                    ps2_pre += torch.dot(buf.view(-1), buf.view(-1))

                if self.consEn == True:
                    # Compare this \pi^2 with what it should have been if the energy was correct
                    ps2_correct = V * ((EoverV**2) - 1.0)

                    # Compute the rescaling factor, only if real
                    if torch.abs(ps2_pre - ps2_correct) < eps1:
                        self.rescaling_pi = 1.0
                    elif ps2_correct < 0.0:
                        self.rescaling_pi = 1.0
                    else:
                        self.rescaling_pi = torch.sqrt(((ps2_correct / (ps2_pre))))

                # Perform the optimization step
                if (self.counter != threshold) and (self.counter0 != threshold0):
                    for p in group["params"]:
                        if p.grad is None:
                            continue
                        d_p = p.grad.data
                        param_state = self.state[p]

                        if "momentum_buffer" not in param_state:
                            buf = param_state["momentum_buffer"] = torch.zeros_like(
                                p.data
                            )
                        else:
                            buf = param_state["momentum_buffer"]

                        # Here the rescaling of momenta to enforce conservation of energy
                        if self.consEn == True:
                            buf.mul_(self.rescaling_pi)

                        buf.add_(-0.5 * dt * (VoverE + EoverV) * d_p)
                        p.data.add_(dt * VoverE * buf)

                    # Updates counters
                    self.counter0 += 1
                    self.counter += 1
                    self.iteration += 1

                    # Checks progress
                    if V < self.min_loss:
                        self.min_loss = V
                        self.counter = 0
                # Bounces
                else:
                    # First we iterate once to compute pi^2, we randomly regenerate the directions, and we compute the new norm squared

                    ps20 = torch.tensor(
                        0.0, device=self.param_groups[0]["params"][0].device
                    )
                    ps2new = torch.tensor(
                        0.0, device=self.param_groups[0]["params"][0].device
                    )

                    for p in group["params"]:
                        param_state = self.state[p]

                        buf = param_state["momentum_buffer"]
                        ps20 += torch.dot(buf.view(-1), buf.view(-1))
                        new_buf = param_state["momentum_buffer"] = (
                            torch.rand(
                                buf.size(), device=buf.device, generator=self.generator
                            )
                            - 0.5
                        )
                        ps2new += torch.dot(new_buf.view(-1), new_buf.view(-1))

                    # Then rescale them
                    for p in group["params"]:
                        param_state = self.state[p]
                        buf = param_state["momentum_buffer"]
                        buf.mul_(torch.sqrt(ps20 / ps2new))

                    # Update counters
                    if self.counter0 == threshold0:
                        self.fixed_bounces_performed += 1
                        if self.fixed_bounces_performed < self.n_fixed_bounces:
                            self.counter0 = 0
                        else:
                            self.counter0 += 1
                    self.counter = 0
        return loss
